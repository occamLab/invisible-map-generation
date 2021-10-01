"""
Contains the FirebaseManager class used for managing the local cache.
"""

import glob
import json
import os
from threading import Semaphore, Thread, Timer
from typing import *
from typing import Dict, Union

import firebase_admin
from firebase_admin import db
from firebase_admin import storage
from varname import nameof


class MapInfo:
    """Container for identifying information for a graph (useful for caching process)

    Attributes:
        map_name (str): Specifies the child of the "maps" database reference to upload the optimized
         graph to; also passed as the map_name argument to the cache_map method
        map_json_name (str): String corresponding to the bucket blob name of the json
        map_dct (dict): String of json containing graph
    """

    def __init__(self, map_name: str, map_json_name: str, map_dct: Dict = None, uid: str = None):
        self.map_name: str = str(map_name)
        self.map_json_blob_name: str = str(map_json_name)
        self.map_dct: Union[dict, str] = dict(map_dct) if map_dct is not None else {}
        self.uid = uid

    def __hash__(self):
        return self.map_json_blob_name.__hash__()

    def __repr__(self):
        return self.map_name


class FirebaseManager:
    """
    Handles Firebase related activities (download and upload) for GraphManager.

    Manages a local cache directory that provides access to

    Class Attributes:
        unprocessed_maps_parent: Simultaneously specifies database reference to listen to in the firebase_listen
         method and the cache location of any maps associated with that database reference.
        processed_upload_to: Simultaneously specifies Firebase bucket path to upload processed graphs to and the
         cache location of processed graphs.
        _app_initialize_dict: Used for initializing the app attribute
        _app: Firebase App initialized with a service account, granting admin privileges. Shared across all instances of
         this class (only initialized once).

    Attributes:
        cache_path: String representing the absolute path to the cache folder. The cache path is evaluated to
         always be located at <path to this file>.cache/
        _bucket: Handle to the Google Cloud Storage bucket
        _db_ref: Database reference representing the node as specified by the GraphManager._unprocessed_listen_to
         class attribute _selected_weights (np.ndarray): Vector selected from the GraphManager._weights_dict
        _listen_kill_timer: Timer that, when expires, exits the firebase listening. Reset every time an event is raised
         by the listener.
        _timer_mutex: Semaphore used in _firebase_get_and_cache_unprocessed_map to only allow one thread to access the
         timer resetting code at a time.
        _max_listen_wait: Amount of time to set the _listen_kill_timer. Any non-positive value results in indefinite
         listening (i.e., the timer not being set).
    """

    _app: Union[firebase_admin.App, None] = None
    _app_initialize_dict: Dict[str, str] = {
        "databaseURL": "https://invisible-map-sandbox.firebaseio.com/",
        "storageBucket": "invisible-map.appspot.com"
    }

    unprocessed_maps_parent: str = "unprocessed_maps"
    processed_upload_to: str = "TestProcessed"

    def __init__(self, firebase_creds: firebase_admin.credentials.Certificate, max_listen_wait: int = -1):
        """
        Args:
            firebase_creds: Firebase credentials
            max_listen_wait: Sets the timer to this amount every time listener produces an event. When the timer
             expires, the firebase listening function exits. If negative, then it listens indefinitely.
        """
        if FirebaseManager._app is None:
            FirebaseManager._app = firebase_admin.initialize_app(firebase_creds, self._app_initialize_dict)

        self._bucket = storage.bucket(app=FirebaseManager._app)
        self._db_ref = db.reference(f'/{self.unprocessed_maps_parent}')
        self.cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.cache")

        # Thread-related attributes for firebase_listen invocation (instantiation here is arbitrary)
        self._listen_kill_timer: Timer = Timer(0, lambda x: x)
        self._firebase_listen_sem: Semaphore = Semaphore()
        self._timer_mutex: Semaphore = Semaphore()
        self._max_listen_wait = max_listen_wait

    def map_info_from_path(self, map_json_path: str) -> Union[MapInfo, None]:
        """
        Parses a json file into a MapInfo instance.

        Args:
            map_json_path: Path to the json file. If the path is not an absolute path, then the cache directory is
             prepended. If this path does not end with ".json", then ".json" is appended.

        Returns:
            MapInfo instance if the specified file exists and is a json file (and None otherwise)
        """
        if not map_json_path.endswith(".json"):
            map_json_path += ".json"
        if not os.path.isabs(map_json_path):
            map_json_path = os.path.join(self.cache_path, map_json_path)

        if not os.path.exists(map_json_path):
            return None

        map_json_path = os.path.join(self.cache_path, map_json_path)
        with open(map_json_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()

        map_json_blob_name = os.path.sep.join(map_json_path.split(os.path.sep)[len(self.cache_path.split(os.path.sep)) + 1:])
        map_dct = json.loads(json_string)
        map_name = self.read_cache_directory(os.path.basename(map_json_blob_name))

        last_folder = map_json_path.split('/')[-2]
        if last_folder == self.unprocessed_maps_parent:
            return MapInfo(map_name, map_json_blob_name, map_dct)
        return MapInfo(map_name, map_json_blob_name, map_dct, last_folder)

    def find_maps(self, pattern: str) -> Set[MapInfo]:
        """
        Returns a set MapInfo objects matching the provided pattern through a recursive search of the unprocessed maps
        cache.

        Args:
            pattern: Pattern to match map file paths in any sub-directory of the cache to.

        Returns:
            Set of matched files as absolute file paths
        """
        matching_filepaths = glob.glob(os.path.join(self.cache_path, os.path.join("**", self.unprocessed_maps_parent,
                                                                                  "**", pattern)), recursive=True)
        matches: Set[MapInfo] = set()
        for match in matching_filepaths:
            if os.path.isdir(match):
                continue
            map_info = self.map_info_from_path(match)
            if isinstance(map_info, MapInfo):
                matches.add(map_info)
        return matches

    def firebase_listen(self, callback: Union[None, Callable], max_wait_override: Union[int, None] = None):
        """
        Wait for and act upon events using the Firebase database reference listener.

        Args:
            max_wait_override: Sets the _max_listen_wait attribute if not none.
            callback: Callback function for when a firebase event occurs (through the listen method). If none is
             provided, then the default map_info_callback of get_map_from_unprocessed_map_event is used.
        """
        if isinstance(max_wait_override, int):
            self._max_listen_wait = max_wait_override
        if self._max_listen_wait <= 0:
            self._db_ref.listen(self.get_map_from_unprocessed_map_event if callback is None else callback)
            return

        self._firebase_listen_sem = Semaphore(0)
        self._timer_mutex = Semaphore(1)
        self._listen_kill_timer = Timer(self._max_listen_wait, self._firebase_listen_sem.release)
        self._listen_kill_timer.start()
        thread_obj = Thread(
            target=lambda: self._db_ref.listen(self.get_map_from_unprocessed_map_event if callback is None else callback))
        thread_obj.start()
        self._firebase_listen_sem.acquire()
        thread_obj.join()
        print("Finished listening to Firebase")

    def read_cache_directory(self, key: str) -> Union[str, None]:
        """Reads the dictionary stored as a json file in <cache folder>/directory.json and returns the value
        associated with the specified key. The key-value pairs in the directory.json map file names to map names.

        Note that no error handling is implemented.

        Args:
            key (str): Key to query the dictionary

        Returns:
            Value associated with the key
        """
        with open(os.path.join(self.cache_path, "directory.json"), "r") as directory_file:
            directory_json = json.loads(directory_file.read())
            directory_file.close()
            loaded = True
        if loaded:
            return directory_json.get(key)
        else:
            return None

    def upload(self, map_info: MapInfo, json_string: str) -> None:
        """Uploads the map json string into the Firebase bucket under the path
        <GraphManager._processed_upload_to>/<processed_map_filename> and updates the appropriate database reference.

        Note that no exception catching is implemented.

        Args:
            map_info (MapInfo): Contains the map name and map json path
            json_string (str): Json string of the map to upload
        """
        processed_map_filename = os.path.basename(map_info.map_json_blob_name)[:-5] + "_processed.json"
        processed_map_full_path = f'{self.processed_upload_to}/{processed_map_filename}'
        print("Attempting to upload {} to the bucket blob {}".format(map_info.map_name, processed_map_full_path))
        processed_map_blob = self._bucket.blob(processed_map_full_path)
        processed_map_blob.upload_from_string(json_string)
        print("Successfully uploaded map data for {}".format(map_info.map_name))
        ref = db.reference("maps")
        if map_info.uid is not None:
            ref = ref.child(map_info.uid)
        ref.child(map_info.map_name).child("map_file").set(processed_map_full_path)
        print("Successfully uploaded database reference maps/{}/map_file to contain the blob path".format(
            map_info.map_name))
        self.cache_map(self.processed_upload_to, map_info, json_string)

    def download_all_maps(self):
        """
        Downloads all maps from Firebase.
        """
        return self._download_all_maps()

    def _download_all_maps(self, map_info: Union[Dict[str, Dict], None] = None, uid: str = None):
        """
        Recursive function for downloading all maps from Firebase.
        """
        if map_info is None:
            map_info = db.reference(self.unprocessed_maps_parent).get()

        for child_key, child_val in map_info.items():
            if isinstance(child_val, str):
                print(f'Downloading {"" if uid is None else uid + "/"}{child_key}')
                self._firebase_get_and_cache_unprocessed_map(child_key, child_val, uid=uid)
            elif isinstance(child_val, dict):
                self._download_all_maps(map_info=child_val, uid=child_key if uid is None else uid)

    def get_map_from_unprocessed_map_event(
            self, event: firebase_admin.db.Event,
            map_info_callback: Union[Callable[[MapInfo], None], None] = None,
            ignore_dict: bool = False
    ) -> None:
        """Acquires MapInfo objects from firebase events corresponding to unprocessed maps.

        Arguments:
            event: A firebase event corresponding a single unprocessed map (event.data is a string) or to a dictionary
             of unprocessed maps (event.data is a dictionary).
            map_info_callback: For every MapInfo object created, invoke this callabck and pass the MapInfo object as the
             argument.
            ignore_dict: If true, no action is taken if event.data is a dictionary.
        """
        if type(event.data) == str:
            # A single new map just got added
            map_info = self._firebase_get_and_cache_unprocessed_map(event.path.lstrip("/"), event.data)
            if map_info_callback is not None and map_info is not None:
                map_info_callback(map_info)
        elif type(event.data) == dict:
            if ignore_dict:
                return
            # This will be a dictionary of all the data that is there initially
            for map_name, map_json in event.data.items():
                if isinstance(map_json, str):
                    map_info = self._firebase_get_and_cache_unprocessed_map(map_name, map_json)
                    if map_info_callback is not None and map_info is not None:
                        map_info_callback(map_info)
                elif isinstance(map_json, dict):
                    for nested_name, nested_json in map_json.items():
                        map_info = self._firebase_get_and_cache_unprocessed_map(nested_name, nested_json, uid=map_name)
                        if map_info_callback is not None and map_info is not None:
                            map_info_callback(map_info)

    def _firebase_get_and_cache_unprocessed_map(self, map_name: str, map_json: str, uid: str = None) \
            -> Union[MapInfo, None]:
        """Acquires a map from the specified blob and caches it.

        A diagnostic message is printed if the map_json_blob_name blob name was not found by Firebase.

        Args:
            map_name (str): Value passed as the map_name argument to the cache_map method; the value of map_name is
             ultimately used for uploading a map to firebase by specifying the child of the 'maps' database reference.
            map_json (str): Value passed as the blob_name argument to the get_blob method of the _bucket
             attribute.

        Returns:
            True if the map was successfully acquired and cached, and false if the map was not found by Firebase
        """
        # Reset the timer
        if self._max_listen_wait > 0:
            self._timer_mutex.acquire()
            self._listen_kill_timer.cancel()
            self._listen_kill_timer = Timer(self._max_listen_wait, self._firebase_listen_sem.release)
            self._listen_kill_timer.start()
            self._timer_mutex.release()

        map_info = MapInfo(map_name, map_json, uid=uid)
        json_blob = self._bucket.get_blob(map_info.map_json_blob_name)
        if json_blob is not None:
            json_data = json_blob.download_as_bytes()
            json_dct = json.loads(json_data)
            map_info.map_dct = json_dct
            self.cache_map(self.unprocessed_maps_parent, map_info, json.dumps(json_dct, indent=2))
            return map_info
        else:
            print("Map '{}' was missing".format(map_info.map_name))
            return None

    def cache_map(self, parent_folder: str, map_info: MapInfo, json_string: str, file_suffix: Union[
            str, None] = None) -> bool:
        """Saves a map to a json file in cache directory.

        Catches any exceptions raised when saving the file (exceptions are raised for invalid arguments) and displays an
        appropriate diagnostic message if one is caught. All of the arguments are checked to ensure that they are, in
        fact strings; if any are not, then a diagnostic message is printed and False is returned.

        Arguments:
            parent_folder (str): Specifies the sub-directory of the cache directory that the map is cached in
            map_info (MapInfo): Contains the map name and map json path in the map_name and map_json_blob_name
             fields respectively. If the last 5 characters of this string do not form the substring ".json",
             then ".json" will be appended automatically.
            json_string (str): The json string that defines the map (this is what is written as the contents of the
             cached map file).
            file_suffix (str): String to append to the file name given by map_info.map_json_blob_name.

        Returns:
            True if map was successfully cached, and False otherwise

        Raises:
            ValueError: Raised if there is any argument (except file_suffix) that is of an incorrect type
            NotADirectoryError: Raised if _resolve_cache_dir method returns false.
        """
        if not isinstance(map_info, MapInfo):
            raise ValueError("Cannot cache map because '{}' argument is not a {} instance"
                             .format(nameof(map_info), nameof(MapInfo)))
        for arg in [parent_folder, map_info.map_name, map_info.map_json_blob_name, json_string]:
            if not isinstance(arg, str):
                raise ValueError("Cannot cache map because '{}' argument is not a string".format(nameof(arg)))

        if not self._resolve_cache_dir():
            raise NotADirectoryError("Cannot cache map because cache folder existence could not be resolved at path {}"
                                     .format(self.cache_path))

        file_suffix_str = (file_suffix if isinstance(file_suffix, str) else "")
        map_json_to_use = str(map_info.map_json_blob_name)
        if len(map_json_to_use) < 6:
            map_json_to_use += file_suffix_str + ".json"
        else:
            if map_json_to_use[-5:] != ".json":
                map_json_to_use += file_suffix_str + ".json"
            else:
                map_json_to_use = map_json_to_use[:-5] + file_suffix_str + ".json"

        cached_file_path = os.path.join(self.cache_path, parent_folder, map_json_to_use)
        try:
            cache_to = os.path.join(parent_folder, map_json_to_use)
            cache_to_split = cache_to.split(os.path.sep)
            cache_to_split_idx = 0
            while cache_to_split_idx < len(cache_to_split) - 1:
                dir_to_check = os.path.join(self.cache_path, os.path.sep.join(cache_to_split[:cache_to_split_idx + 1]))
                if not os.path.exists(dir_to_check):
                    os.mkdir(dir_to_check)
                cache_to_split_idx += 1

            with open(cached_file_path, "w") as map_json_file:
                map_json_file.write(json_string)
                map_json_file.close()

            self._append_to_cache_directory(os.path.basename(map_json_to_use), map_info.map_name)
            print("Successfully cached {}".format(cached_file_path))
            return True
        except Exception as ex:
            print("Could not cache map {} due to error: {}".format(map_json_to_use, ex))
            return False

    def _resolve_cache_dir(self) -> bool:
        """Returns true if the cache folder exists, and attempts to create a new one if there is none.

        A file named directory.json is also created in the cache folder.

        This method catches all exceptions associated with creating new directories/files and displays a corresponding
        diagnostic message.

        Returns:
            True if no exceptions were caught and False otherwise
        """
        if not os.path.exists(self.cache_path):
            try:
                os.mkdir(self.cache_path)
            except Exception as ex:
                print("Could not create a cache directory at {} due to error: {}".format(self.cache_path, ex))
                return False

        directory_path = os.path.join(self.cache_path, "directory.json")
        if not os.path.exists(directory_path):
            try:
                with open(os.path.join(self.cache_path, "directory.json"), "w") as directory_file:
                    directory_file.write(json.dumps({}))
                    directory_file.close()
                return True
            except Exception as ex:
                print("Could not create {} file due to error: {}".format(directory_path, ex))
        else:
            return True

    def _append_to_cache_directory(self, key: str, value: str) -> None:
        """Appends the specified key-value pair to the dictionary stored as a json file in
        <cache folder>/directory.json.

        If the key already exists in the dictionary, its value is overwritten. Note that no error handling is
        implemented.

        Args:
            key (str): Key to store value in
            value (str): Value to store under key
        """
        directory_json_path = os.path.join(self.cache_path, "directory.json")
        with open(directory_json_path, "r") as directory_file_read:
            directory_json = json.loads(directory_file_read.read())
            directory_file_read.close()
        directory_json[key] = value
        new_directory_json = json.dumps(directory_json, indent=2)
        with open(directory_json_path, "w") as directory_file_write:
            directory_file_write.write(new_directory_json)
            directory_file_write.close()
