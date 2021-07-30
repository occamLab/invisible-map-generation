"""
Script that runs on the deepthought.olin.edu server to listen to Firebase for
new maps and process and upload them.

Author: Allison Li
"""
import json
import os
from typing import *
from threading import Semaphore, Thread, Timer

import firebase_admin
from firebase_admin import db
from firebase_admin import storage
from varname import nameof

from map_processing.graph_utils import MapInfo


class FirebaseManager:
    """
    Handles Firebase related activities (download and upload) for GraphManager
    """
    _app_initialize_dict: Dict[str, str] = {
        "databaseURL": "https://invisible-map-sandbox.firebaseio.com/",
        "storageBucket": "invisible-map.appspot.com"
    }
    _initialized_app: bool = False

    unprocessed_listen_to: str = "unprocessed_maps"
    processed_upload_to: str = "TestProcessed"

    def __init__(self, firebase_creds: firebase_admin.credentials.Certificate, download_only: bool = False,
                 max_wait: int = -1):
        if not self._initialized_app:
            self._app = firebase_admin.initialize_app(firebase_creds, self._app_initialize_dict)
            self._initialized_app = True

        self._download_only = download_only
        self._max_wait = max_wait

        self._bucket = storage.bucket(app=self._app)
        self._db_ref = db.reference(f'/{self.unprocessed_listen_to}')
        self.cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.cache")

        # Thread-related attributes for firebase_listen invocation (instantiation here is arbitrary)
        self._listen_kill_timer: Timer = Timer(0, lambda x: x)
        self._firebase_listen_sem: Semaphore = Semaphore()
        self._timer_mutex: Semaphore = Semaphore()
        self._firebase_listen_max_wait: int = 0

    def set_listener(self, listener):
        self._db_ref.listen(listener)

    def firebase_listen(self):
        if self._max_wait == -1:
            self.set_listener(self.get_map_from_event)
        else:
            self._firebase_listen_max_wait = self._max_wait
            self._firebase_listen_sem = Semaphore(0)
            self._timer_mutex = Semaphore(1)
            self._listen_kill_timer = Timer(self._firebase_listen_max_wait, self._firebase_listen_sem.release)
            self._listen_kill_timer.start()
            thread_obj = Thread(target=lambda: self.set_listener(self.get_map_from_event))
            thread_obj.start()
            self._firebase_listen_sem.acquire()
            thread_obj.join()
            print("Finished listening to Firebase")

    def upload(self, map_info: MapInfo, json_string: str) -> None:
        """Uploads the map json string into the Firebase bucket under the path
        <GraphManager._processed_upload_to>/<processed_map_filename> and updates the appropriate database reference.

        Note that no exception catching is implemented.

        Args:
            map_info (MapInfo): Contains the map name and map json path
            json_string (str): Json string of the map to upload
        """
        if not self._download_only:
            processed_map_filename = os.path.basename(map_info.map_json)[:-5] + "_processed.json"
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
            self._cache_map(self.processed_upload_to, map_info, json_string)

    def get_map_from_event(self, event) -> MapInfo:
        if type(event.data) == str:
            # A single new map just got added
            return self._firebase_get_unprocessed_map(event.path.lstrip("/"), event.data)
        elif type(event.data) == dict:
            # This will be a dictionary of all the data that is there initially
            for map_name, map_json in event.data.items():
                if isinstance(map_json, str):
                    return self._firebase_get_unprocessed_map(map_name, map_json)
                elif isinstance(map_json, dict):
                    for nested_name, nested_json in map_json.items():
                        return self._firebase_get_unprocessed_map(nested_name, nested_json, uid=map_name)

    def _firebase_get_unprocessed_map(self, map_name: str, map_json: str, uid: str = None) -> MapInfo:
        """Acquires a map from the specified blob and caches it.

        A diagnostic message is printed if the map_json blob name was not found by Firebase.

        Args:
            map_name (str): Value passed as the map_name argument to the _cache_map method; the value of map_name is
             ultimately used for uploading a map to firebase by specifying the child of the 'maps' database reference.
            map_json (str): Value passed as the blob_name argument to the get_blob method of the _bucket
             attribute.

        Returns:
            True if the map was successfully acquired and cached, and false if the map was not found by Firebase
        """
        # Reset the timer
        if self._max_wait != -1:
            self._timer_mutex.acquire()
            self._listen_kill_timer.cancel()
            self._listen_kill_timer = Timer(self._firebase_listen_max_wait, self._firebase_listen_sem.release)
            self._listen_kill_timer.start()
            self._timer_mutex.release()

        map_info = MapInfo(map_name, map_json, uid=uid)
        json_blob = self._bucket.get_blob(map_info.map_json)
        if json_blob is not None:
            json_data = json_blob.download_as_bytes()
            json_dct = json.loads(json_data)
            map_info.map_dct = json_dct
            self._cache_map(self.unprocessed_listen_to, map_info, json.dumps(json_dct, indent=2))
            return map_info
        else:
            print("Map '{}' was missing".format(map_info.map_name))
            return None

    def _cache_map(self, parent_folder: str, map_info: MapInfo, json_string: str, file_suffix: Union[
            str, None] = None) -> bool:
        """Saves a map to a json file in cache directory.

        Catches any exceptions raised when saving the file (exceptions are raised for invalid arguments) and displays an
        appropriate diagnostic message if one is caught. All of the arguments are checked to ensure that they are, in
        fact strings; if any are not, then a diagnostic message is printed and False is returned.

        Arguments:
            parent_folder (str): Specifies the sub-directory of the cache directory that the map is cached in
            map_info (MapInfo): Contains the map name and map json path in the map_name and map_json
             fields respectively. If the last 5 characters of this string do not form the substring ".json",
             then ".json" will be appended automatically.
            json_string (str): The json string that defines the map (this is what is written as the contents of the
             cached map file).
            file_suffix (str): String to append to the file name given by map_info.map_json.

        Returns:
            True if map was successfully cached, and False otherwise

        Raises:
            ValueError: Raised if there is any argument (except file_suffix) that is of an incorrect type
            NotADirectoryError: Raised if _resolve_cache_dir method returns false.
        """
        if not isinstance(map_info, MapInfo):
            raise ValueError("Cannot cache map because '{}' argument is not a {} instance"
                             .format(nameof(map_info), nameof(MapInfo)))
        for arg in [parent_folder, map_info.map_name, map_info.map_json, json_string]:
            if not isinstance(arg, str):
                raise ValueError("Cannot cache map because '{}' argument is not a string".format(nameof(arg)))

        if not self._resolve_cache_dir():
            raise NotADirectoryError("Cannot cache map because cache folder existence could not be resolved at path {}"
                                     .format(self.cache_path))

        file_suffix_str = (file_suffix if isinstance(file_suffix, str) else "")
        map_json_to_use = str(map_info.map_json)
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
