"""
Contains the CacheManagerSingleton class used for managing the local cache.
"""

import uuid
import glob
import json
import os
import random
from threading import Semaphore, Thread, Timer
from typing import Dict, Union, List, Optional, Set, Callable
from collections import defaultdict


import firebase_admin
import numpy as np
from firebase_admin import db, storage
from varname import nameof

from map_processing import GT_TAG_DATASETS, GROUND_TRUTH_MAPPING_STARTING_PT
from map_processing.data_models import (
    GTDataSet,
    OSweepResults,
    OMultiSweepResult,
    OResultPseudoGTMetricValidation,
)

from dotenv import load_dotenv

load_dotenv()


class MapInfo:
    """Container for identifying information for a graph (useful for caching process)

    Attributes:
        map_name (str): Specifies the child of the "maps" database reference to upload the optimized
         graph to; also passed as the map_name argument to the cache_map method
        map_json_name (str): String corresponding to the __bucket blob name of the json
        map_dct (dict): String of json containing graph
    """

    def __init__(
        self,
        map_name: str,
        map_json_name: str,
        map_dct: Dict = None,
        uid: str = None,
        map_bounds: Dict = None,
    ):
        self.map_name: str = str(map_name)
        self.map_json_blob_name: str = str(map_json_name)
        self.map_dct: Union[dict, str] = dict(map_dct) if map_dct is not None else {}
        self.uid = uid
        self.map_bounds: dict = map_bounds

    def __hash__(self):
        return self.map_json_blob_name.__hash__()

    def __repr__(self):
        return self.map_name


class CacheManagerSingleton:
    """
    Handles caching for graphs (primarily though downloading/uploading to/from Firebase and local
    caching of synthetically generated graphs).

    Notes:
        Implemented as a singleton

    Class Attributes:
        UNPROCESSED_MAPS_PARENT: Simultaneously specifies database reference to listen to in the
            firebase_listen method and the cache location of any maps associated with that database
            reference.
        PROCESSED_UPLOAD_TO: Simultaneously specifies Firebase bucket path to upload processed
            graphs to and the cache location of processed graphs.
        __app_initialize_dict: Used for initializing the app attribute
        CACHE_PATH: String representing the absolute path to the cache folder. The cache path is
            evaluated to always be located at <path to this file>.cache/

    Attributes:
        __app: Firebase App initialized with a service account, granting admin privileges. Shared
            across all instances of this class (only initialized once).
        __bucket: Handle to the Google Cloud Storage __bucket
        __db_ref: Database reference representing the node as specified by the
            GraphManager._unprocessed_listen_to
        class attribute selected_weights (np.ndarray): Vector selected from the
            GraphManager.WEIGHTS_DICT
        __listen_kill_timer: Timer that, when expires, exits the firebase listening. Reset every
            time an event is raised by the listener.
        __timer_mutex: Semaphore used in _firebase_get_and_cache_unprocessed_map to only allow one
            thread to access the timer resetting code at a time.
        __max_listen_wait: Amount of time to set the _listen_kill_timer. Any non-positive value
            results in indefinite listening (i.e., the timer not being set).
    """

    __instance = None
    __app_initialize_dict: Dict[str, str] = {
        "databaseURL": "https://stepnavigation-default-rtdb.firebaseio.com/",
        "storageBucket": "stepnavigation.appspot.com",
    }

    UNPROCESSED_MAPS_PARENT: str = "unprocessed_maps"
    GENERATED_MAPS_PARENTS: str = "generated"
    PROCESSED_UPLOAD_TO: str = "protobuf"
    SWEEP_PROCESSED_UPLOAD_TO: str = "SweepProcessed"
    GROUND_TRUTH_PARENT: str = "ground_truth"
    SWEEP_RESULTS_PARENT: str = "sweep_results"
    PGT_VALIDATION_RESULTS_PARENT: str = "pgt_validation_results"
    GROUND_TRUTH_MAPPING_FILE_NAME = "ground_truth_mapping.json"
    FIREBASE_CONFIG_FILE_NAME = "firebase_device_config.json"
    CONFIG_PATH = "configs"

    CACHE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.cache")
    GROUND_TRUTH_PATH = os.path.join(CACHE_PATH, GROUND_TRUTH_PARENT)
    GROUND_TRUTH_MAPPING_PATH = os.path.join(
        CONFIG_PATH, GROUND_TRUTH_MAPPING_FILE_NAME
    )
    SWEEP_RESULTS_PATH: str = os.path.join(CACHE_PATH, SWEEP_RESULTS_PARENT)
    PGT_VALIDATION_RESULTS_PATH: str = os.path.join(
        CACHE_PATH, PGT_VALIDATION_RESULTS_PARENT
    )
    FIREBASE_CONFIG_PATH = os.path.join(CONFIG_PATH, FIREBASE_CONFIG_FILE_NAME)

    def __init__(
        self,
        firebase_creds: Optional[firebase_admin.credentials.Certificate] = None,
        max_listen_wait: int = -1,
    ):
        """
        Args:
            firebase_creds: Firebase credentials. If not set in the initializer, then later
                functionality that depends on communication with Firebase will break until the
                credentials are
            max_listen_wait: Sets the timer to this amount every time listener produces an event.
                When the timer expires, the firebase listening function exits. If negative, then it
                listens indefinitely.
        """
        self.__synch_mutex: Semaphore = Semaphore()

        self.__app: Optional[firebase_admin.App] = None
        self.__bucket: Optional[firebase_admin.storage.storage.Bucket] = None
        self.__db_ref: Optional[db.Reference] = None
        self.__were_credentials_set: bool = False
        if firebase_creds is not None:
            self.set_credentials(firebase_creds)

        # Thread-related attributes for firebase_listen invocation (instantiation here is arbitrary)
        self.__listen_kill_timer: Timer = Timer(0, lambda x: x)
        self.__firebase_listen_sem: Semaphore = Semaphore()
        self.__timer_mutex: Semaphore = Semaphore()
        self.__max_listen_wait = max_listen_wait

    def __new__(cls, *args, **kwargs):
        """Implements the singleton pattern."""
        if cls.__instance is None:
            cls.__instance = super(CacheManagerSingleton, cls).__new__(cls)
            # Generate all ground truth data files from hard-coded data
            CacheManagerSingleton.export_all_ground_truth_data()
        return cls.__instance

    @property
    def were_credentials_set(self) -> bool:
        return self.__were_credentials_set

    def firebase_listen(
        self,
        callback: Union[None, Callable],
        max_wait_override: Union[int, None] = None,
    ):
        """Wait for and act upon events using the Firebase database reference listener.

        Notes:
            Acquires the __synch_mutex (calling from another thread will block until this completes).

        Args:
            max_wait_override: Sets the __max_listen_wait attribute if not none.
            callback: Callback function for when a firebase event occurs (through the listen
                method). If none is provided, then the default map_info_callback of
                get_map_from_unprocessed_map_event is used.
        """
        with self.__synch_mutex:
            if isinstance(max_wait_override, int):
                self.__max_listen_wait = max_wait_override
            if self.__max_listen_wait <= 0:
                self.__db_ref.listen(
                    self.get_map_from_unprocessed_map_event
                    if callback is None
                    else callback
                )
                return

            self.__firebase_listen_sem = Semaphore(0)
            self.__timer_mutex = Semaphore(1)
            self.__listen_kill_timer = Timer(
                self.__max_listen_wait, self.__firebase_listen_sem.release
            )
            self.__listen_kill_timer.start()
            thread_obj = Thread(
                target=lambda: self.__db_ref.listen(
                    self.get_map_from_unprocessed_map_event
                    if callback is None
                    else callback
                )
            )
            thread_obj.start()
            self.__firebase_listen_sem.acquire()
            thread_obj.join()

    def upload(
        self, map_info: MapInfo, proto_bytes: bytes, verbose: bool = False
    ) -> None:
        """Uploads the map proto bytes into the Firebase __bucket under the path
        <GraphManager._processed_upload_to>/<processed_map_filename> and updates the appropriate database reference.

        Notes:
            - Acquires the __synch_mutex (calling from another thread will block until this
              completes).
            - Note that no exception catching is implemented.

        Args:
            map_info (MapInfo): Contains the map name and map json path
            proto_bytes (bytes): Protobuf bytes of the map to upload
            verbose: TODO
        """
        with self.__synch_mutex:
            processed_map_filename = (
                os.path.basename(map_info.map_json_blob_name)[:-5] + "_processed.proto"
            )
            processed_map_full_path = (
                f"{self.PROCESSED_UPLOAD_TO}/{processed_map_filename}"
            )
            if verbose:
                print(
                    f"Attempting to upload {map_info.map_name} to the __bucket blob \
                     {processed_map_full_path}"
                )

            processed_map_blob = self.__bucket.blob(processed_map_full_path)
            token = uuid.uuid4()
            processed_map_blob.metadata = {"firebaseStorageDownloadTokens": token}
            processed_map_blob.upload_from_string(proto_bytes)

            ref = db.reference("maps")
            if map_info.uid is not None:
                ref = ref.child(map_info.uid)
            ref.child(map_info.map_name).child("map_file").set(processed_map_full_path)
            ref.child("latestProcessedMap").set(processed_map_full_path)

            if verbose:
                print(
                    f"Successfully uploaded database reference maps/{map_info.map_name}/"
                    f"map_file to contain the blob path"
                )
            CacheManagerSingleton.cache_map(
                CacheManagerSingleton.PROCESSED_UPLOAD_TO,
                map_info,
                proto_bytes,
                verbose=verbose,
            )

    def download_all_maps(self):
        """Downloads all maps from Firebase."""
        return self._download_all_maps_recur()

    def combine_shared_maps(self, map_seed: Union[str, MapInfo]):
        """Combines Maps with shared CAs then caches it"""
        map_info = CacheManagerSingleton.flatten_firebase(
            tree=db.reference(self.UNPROCESSED_MAPS_PARENT).get()
        )
        if isinstance(map_seed, str):
            combined_map = self._combine_maps_with_shared_cloud_anchors(
                map_info, map_seed
            )
            map_name = map_seed.split(".")[0]
        if isinstance(map_seed, MapInfo):
            combined_map = self._combine_maps_with_shared_cloud_anchors(
                map_info, map_seed=map_seed
            )
            map_name = map_seed.map_name
        combined_map.map_name = f"{map_name}_combined"
        combined_map.map_json_blob_name = f"fixedtesting/{map_name}_combined"
        CacheManagerSingleton.cache_json_map(
            CacheManagerSingleton.UNPROCESSED_MAPS_PARENT,
            combined_map,
            json.dumps(combined_map.map_dct, indent=2),
        )
        return combined_map

    @staticmethod
    def combine_maps(
        map_seed: MapInfo,
        new_map: MapInfo,
    ) -> MapInfo:
        """
        Combine maps from Firebase

        Notes:
        Combines the dictionaries from both maps into one, then reindexes it.
        map_bounds indicates the index for the end of a map.
        Alters the coordinate plane of new_map to the coordinate plane of map_seed

        Args:
        map_seed: A MapInfo instance that serves as the map to be added to
        new_map:  A MapInfo instance that is the map to be added

        Return:
        complete_map: A MapInfo instance that contains the combined map_dictionary from both maps
        """
        map_dictionary = defaultdict(list)
        id_len = 0
        map_bounds = {}
        anchor_info = {}
        matching_map = set([map_seed, new_map])
        map_name = ""

        for map_set in matching_map:
            if map_set is None:
                continue
            map_json_name = map_set.map_json_blob_name
            map_name += map_set.map_name
            anchor_info[map_set.map_name] = {}
            for pose_data in map_set.map_dct["pose_data"]:
                pose_data["id"] += id_len
            for cloud_data in map_set.map_dct["cloud_data"]:
                for instance in cloud_data:
                    instance["poseId"] += id_len
                    anchor_info[map_set.map_name][
                        instance["cloudIdentifier"]
                    ] = instance["pose"]
            for key, values in map_set.map_dct.items():
                map_dictionary[key].extend(values)
            id_len += len(map_set.map_dct["pose_data"])
            map_bounds[map_set.map_name] = id_len

        map_dictionary["map_id"] = "combined_map"
        map_bounds = dict(sorted(map_bounds.items(), key=lambda item: item[1]))

        if len(anchor_info) > 1:
            all_anchor_ids = [set(anchor_info[anchor].keys()) for anchor in anchor_info]
            intersect = set.intersection(*all_anchor_ids)
            anchor_id = random.choice(list(intersect))

            anchor_positions = {
                map_id: np.transpose(np.reshape(anchor_info[map_id][anchor_id], (4, 4)))
                for map_id in anchor_info
            }

            for pose_data in map_dictionary["pose_data"]:
                for map_data in map_bounds:
                    if pose_data["id"] < map_bounds[map_data]:
                        fixed = np.linalg.inv(anchor_positions[map_data]).dot(
                            np.transpose(np.reshape(pose_data["pose"], (4, 4)))
                        )
                        pose_data["pose"] = list(
                            np.reshape(np.transpose(fixed), (1, 16))[0]
                        )
                        break

            for cloud_data in map_dictionary["cloud_data"]:
                for instance in cloud_data:
                    for map_data in map_bounds:
                        if instance["poseId"] < map_bounds[map_data]:
                            fixed = np.linalg.inv(anchor_positions[map_data]).dot(
                                np.transpose(np.reshape(instance["pose"], (4, 4)))
                            )
                            instance["pose"] = list(
                                np.reshape(np.transpose(fixed), (1, 16))[0]
                            )
                            break

        complete_map = MapInfo(
            map_name=map_name,
            map_dct=map_dictionary,
            map_json_name=map_json_name,
            map_bounds=map_bounds,
        )
        return complete_map

    def get_map_from_unprocessed_map_event(
        self,
        event: firebase_admin.db.Event,
        map_info_callback: Union[Callable[[MapInfo], None], None] = None,
        ignore_dict: bool = False,
        override_all: bool = False,
    ) -> None:
        """Acquires MapInfo objects from firebase events corresponding to unprocessed maps.

        Arguments:
            event: A firebase event corresponding a single unprocessed map (event.data is a string)
                or to a dictionary of unprocessed maps (event.data is a dictionary).
            map_info_callback: For every MapInfo object created, invoke this callback and pass the
                MapInfo object as the argument.
            ignore_dict: If true, no action is taken if `event.data` is a dictionary.
            override_all: If true, reprocess every map in firebase.
        """
        firebase_reference = db.reference("unprocessed_maps")
        if isinstance(event.data, str):
            # A single new map just got added
            map_info = self._firebase_get_and_cache_unprocessed_map(
                event.path.lstrip("/"), event.data
            )
            if (
                "map_file" in firebase_reference.child(map_info.map_name).get()
                and not override_all
            ):
                return
            if map_info_callback is not None and map_info is not None:
                map_info_callback(map_info)
        elif isinstance(event.data, dict):
            if ignore_dict:
                return
            # This will be a dictionary of all the data that is there initially
            for map_name, map_json in event.data.items():
                if isinstance(map_json, str):
                    map_info = self._firebase_get_and_cache_unprocessed_map(
                        map_name, map_json
                    )
                    if (
                        firebase_reference.child(map_info.map_name).get() is not None
                        and "map_file"
                        in firebase_reference.child(map_info.map_name).get()
                        and not override_all
                    ):
                        continue
                    if map_info_callback is not None:
                        map_info_callback(map_info)
                elif isinstance(map_json, dict):
                    for nested_name, nested_json in map_json.items():
                        map_info = self._firebase_get_and_cache_unprocessed_map(
                            nested_name, nested_json, uid=map_name
                        )
                        if map_info is None:
                            continue
                        if (
                            firebase_reference.child(map_name)
                            .child(map_info.map_name)
                            .get()
                            is not None
                            and "map_file"
                            in firebase_reference.child(map_name)
                            .child(map_info.map_name)
                            .get()
                            and not override_all
                        ):
                            continue
                        if map_info_callback is not None:
                            map_info_callback(map_info)

    @staticmethod
    def map_info_from_path(map_json_path: str) -> Union[MapInfo, None]:
        """
        Parses a json file into a MapInfo instance.

        Args:
            map_json_path: Path to the json file. If the path is not an absolute path, then the
                cache directory is prepended. If this path does not end with ".json", then ".json"
                is appended.

        Returns:
            MapInfo instance if the specified file exists and is a json file (and None otherwise)
        """
        if not map_json_path.endswith(".json"):
            map_json_path += ".json"
        if not os.path.isabs(map_json_path):
            map_json_path = os.path.join(
                CacheManagerSingleton.CACHE_PATH, map_json_path
            )

        if not os.path.exists(map_json_path):
            return None

        map_json_path = os.path.join(CacheManagerSingleton.CACHE_PATH, map_json_path)
        with open(map_json_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()

        map_json_blob_name = os.path.sep.join(
            map_json_path.split(os.path.sep)[
                len(CacheManagerSingleton.CACHE_PATH.split(os.path.sep)) + 1 :
            ]
        )
        map_dct = json.loads(json_string)
        map_name = CacheManagerSingleton._read_cache_directory(
            os.path.basename(map_json_blob_name)
        )

        last_folder = map_json_path.split("/")[-2]
        if last_folder == CacheManagerSingleton.UNPROCESSED_MAPS_PARENT:
            return MapInfo(map_name, map_json_blob_name, map_dct)
        return MapInfo(map_name, map_json_blob_name, map_dct, last_folder)

    @staticmethod
    def find_maps(
        pattern: str, search_restriction: int = 0, paths: bool = False
    ) -> Set[MapInfo]:
        """Returns a set MapInfo objects matching the provided pattern through a recursive search of the cache
        directory.

        Notes:
            Prepends "**" to the pattern and calls `glob.glob` with recursive=True

        Args:
            pattern: Pattern to match map file paths in any subdirectory of the cache to.
            search_restriction: Determines which directory to search within.
                0: UNPROCESSED_MAPS_PARENT
                1: Entire cache folder
                2: GENERATED_MAPS_PARENT

        Returns:
            Set of matched files as absolute file paths
        """
        recursive = True

        search_dirs = {
            0: CacheManagerSingleton.UNPROCESSED_MAPS_PARENT,
            1: "",
            2: CacheManagerSingleton.GENERATED_MAPS_PARENTS,
        }

        matching_filepaths = glob.glob(
            os.path.join(
                CacheManagerSingleton.CACHE_PATH,
                os.path.join(
                    search_dirs[search_restriction],
                    "**",
                    pattern,
                ),
            ),
            recursive=recursive,
        )

        if paths:
            return matching_filepaths

        matches: Set[MapInfo] = set()
        for match in matching_filepaths:
            if os.path.isdir(match):
                continue
            map_info = CacheManagerSingleton.map_info_from_path(match)
            if isinstance(map_info, MapInfo):
                matches.add(map_info)
        return matches

    @staticmethod
    def cache_json_map(
        parent_folder: str,
        map_info: MapInfo,
        json_string: str,
        file_suffix: Union[str, None] = None,
        verbose: bool = False,
    ) -> bool:
        """Saves a map to a json file in cache directory.

        Catches any exceptions raised when saving the file (exceptions are raised for invalid arguments) and displays an
        appropriate diagnostic message if one is caught. All the arguments are checked to ensure that they are, in
        fact strings; if any are not, then a diagnostic message is printed and False is returned.

        Arguments:
            parent_folder (str): Specifies the subdirectory of the cache directory that the map is cached in
            map_info (MapInfo): Contains the map name and map json path in the map_name and map_json_blob_name
             fields respectively. If the last 5 characters of this string do not form the substring ".json",
             then ".json" will be appended automatically.
            json_string (str): The json string that defines the map (this is what is written as the contents of the
             cached map file).
            file_suffix (str): String to append to the file name given by map_info.map_json_blob_name.
            verbose: TODO

        Returns:
            True if map was successfully cached, and False otherwise

        Raises:
            ValueError: Raised if there is any argument (except file_suffix) that is of an incorrect type
            NotADirectoryError: Raised if _resolve_cache_dir method returns false.
        """
        if not isinstance(map_info, MapInfo):
            raise ValueError(
                "Cannot cache map because '{}' argument is not a {} instance".format(
                    nameof(map_info), nameof(MapInfo)
                )
            )
        for arg in [
            parent_folder,
            map_info.map_name,
            map_info.map_json_blob_name,
            json_string,
        ]:
            if not isinstance(arg, str):
                raise ValueError(
                    "Cannot cache map because '{}' argument is not a string".format(
                        nameof(arg)
                    )
                )

        if not CacheManagerSingleton._resolve_cache_dir():
            raise NotADirectoryError(
                "Cannot cache map because cache folder existence could not be resolved at path {}".format(
                    CacheManagerSingleton.CACHE_PATH
                )
            )

        file_suffix_str = file_suffix if isinstance(file_suffix, str) else ""
        map_json_to_use = str(map_info.map_json_blob_name)
        if len(map_json_to_use) < 6:
            map_json_to_use += file_suffix_str + ".json"
        else:
            if map_json_to_use[-5:] != ".json":
                map_json_to_use += file_suffix_str + ".json"
            else:
                map_json_to_use = map_json_to_use[:-5] + file_suffix_str + ".json"

        cached_file_path = os.path.join(
            CacheManagerSingleton.CACHE_PATH, parent_folder, map_json_to_use
        )
        try:
            cache_to = os.path.join(parent_folder, map_json_to_use)
            cache_to_split = cache_to.split(os.path.sep)
            cache_to_split_idx = 0
            while cache_to_split_idx < len(cache_to_split) - 1:
                dir_to_check = os.path.join(
                    CacheManagerSingleton.CACHE_PATH,
                    os.path.sep.join(cache_to_split[: cache_to_split_idx + 1]),
                )
                if not os.path.exists(dir_to_check):
                    os.mkdir(dir_to_check)
                cache_to_split_idx += 1

            with open(cached_file_path, "w") as map_json_file:
                map_json_file.write(json_string)
                map_json_file.close()

            CacheManagerSingleton._append_to_cache_directory(
                os.path.basename(map_json_to_use), map_info.map_name
            )

            if verbose:
                print("Successfully cached {}".format(cached_file_path))
            return True
        except Exception as ex:
            if verbose:
                print(
                    "Could not cache map {} due to error: {}".format(
                        map_json_to_use, ex
                    )
                )
            return False

    @staticmethod
    def cache_map(
        parent_folder: str,
        map_info: MapInfo,
        proto_bytes: bytes,
        file_suffix: Union[str, None] = None,
        verbose: bool = False,
    ) -> bool:
        """Saves a map to a json file in cache directory.

        Catches any exceptions raised when saving the file (exceptions are raised for invalid arguments) and displays an
        appropriate diagnostic message if one is caught. All the arguments are checked to ensure that they are, in
        fact strings; if any are not, then a diagnostic message is printed and False is returned.

        Arguments:
            parent_folder (str): Specifies the subdirectory of the cache directory that the map is cached in
            map_info (MapInfo): Contains the map name and map json path in the map_name and map_json_blob_name
             fields respectively. If the last 5 characters of this string do not form the substring ".json",
             then ".json" will be appended automatically.
            proto_bytes (bytes): The protobuf bytes that defines the map (this is what is written as the contents of the
             cached map file).
            file_suffix (str): String to append to the file name given by map_info.map_json_blob_name.
            verbose: TODO

        Returns:
            True if map was successfully cached, and False otherwise

        Raises:
            ValueError: Raised if there is any argument (except file_suffix) that is of an incorrect type
            NotADirectoryError: Raised if _resolve_cache_dir method returns false.
        """
        if not isinstance(map_info, MapInfo):
            raise ValueError(
                "Cannot cache map because '{}' argument is not a {} instance".format(
                    nameof(map_info), nameof(MapInfo)
                )
            )
        for arg in [
            parent_folder,
            map_info.map_name,
            map_info.map_json_blob_name,
        ]:
            if not isinstance(arg, str):
                raise ValueError(
                    "Cannot cache map because '{}' argument is not a string".format(
                        nameof(arg)
                    )
                )
        if not isinstance(proto_bytes, bytes):
            raise ValueError(
                "Cannot cache map because '{}' argument is not in bytes".format(
                    nameof(proto_bytes)
                )
            )
        if not CacheManagerSingleton._resolve_cache_dir():
            raise NotADirectoryError(
                "Cannot cache map because cache folder existence could not be resolved at path {}".format(
                    CacheManagerSingleton.CACHE_PATH
                )
            )

        file_suffix_str = file_suffix if isinstance(file_suffix, str) else ""
        map_proto_to_use = str(map_info.map_json_blob_name)
        if len(map_proto_to_use) < 6:
            map_proto_to_use += file_suffix_str + ".proto"
        else:
            if map_proto_to_use[-5:] != ".proto":
                map_proto_to_use += file_suffix_str + ".proto"
            else:
                map_proto_to_use = map_proto_to_use[:-5] + file_suffix_str + ".proto"

        cached_file_path = os.path.join(
            CacheManagerSingleton.CACHE_PATH, parent_folder, map_proto_to_use
        )
        try:
            cache_to = os.path.join(parent_folder, map_proto_to_use)
            cache_to_split = cache_to.split(os.path.sep)
            cache_to_split_idx = 0
            while cache_to_split_idx < len(cache_to_split) - 1:
                dir_to_check = os.path.join(
                    CacheManagerSingleton.CACHE_PATH,
                    os.path.sep.join(cache_to_split[: cache_to_split_idx + 1]),
                )
                if not os.path.exists(dir_to_check):
                    os.mkdir(dir_to_check)
                cache_to_split_idx += 1

            with open(cached_file_path, "w") as map_json_file:
                map_json_file.write(proto_bytes)
                map_json_file.close()

            CacheManagerSingleton._append_to_cache_directory(
                os.path.basename(map_proto_to_use), map_info.map_name
            )

            if verbose:
                print("Successfully cached {}".format(cached_file_path))
            return True
        except Exception as ex:
            if verbose:
                print(
                    "Could not cache map {} due to error: {}".format(
                        map_proto_to_use, ex
                    )
                )
            return False

    @staticmethod
    def export_all_ground_truth_data():
        for dataset_name, dataset in GT_TAG_DATASETS.items():
            CacheManagerSingleton.cache_ground_truth_data(
                gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(dataset),
                dataset_name=dataset_name,
            )

    @staticmethod
    def export_all_ground_truth_data():
        for dataset_name, dataset in GT_TAG_DATASETS.items():
            CacheManagerSingleton.cache_ground_truth_data(
                gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(dataset),
                dataset_name=dataset_name,
            )

    @staticmethod
    def find_ground_truth_data_from_map_info(map_info: List[MapInfo]) -> Optional[Dict]:
        """Uses the ground truth mapping to find the dataset matching the map_info object.

        Args:
            map_info: Specifies the ground truth dataset to look for with its map_name attribute.

        Returns:
            None if there is not exactly 1 match found; if there is exactly 1 match found, the deserialized object
             of the ground truth data is parsed and returned as a dictionary that maps tag IDs to their poses as
             7-element vectors.
        """
        matching_datasets = []
        gt_mapping_dict: dict
        if os.path.exists(CacheManagerSingleton.GROUND_TRUTH_MAPPING_PATH):
            with open(CacheManagerSingleton.GROUND_TRUTH_MAPPING_PATH, "r") as f:
                gt_mapping_dict = json.load(f)
        else:
            gt_mapping_dict = GROUND_TRUTH_MAPPING_STARTING_PT
        for map_data in map_info:
            for item in gt_mapping_dict.items():
                for map_name in item[1]:
                    if map_data.map_name == map_name:
                        matching_datasets.append(item[0])
        if len(matching_datasets) != 1:
            return None
        else:
            ret = {}
            unprocessed = (
                CacheManagerSingleton.find_ground_truth_data_from_dataset_name(
                    matching_datasets[0]
                )
            )
            for pose_dict in unprocessed["poses"]:
                ret[pose_dict["tag_id"]] = np.array(pose_dict["pose"])
            return ret

    @staticmethod
    def find_ground_truth_data_from_dataset_name(dataset_name: str) -> Optional[Dict]:
        """Look for a ground truth dataset stored in the ground truth directory of the cache that matches the dataset
        name. Specifically, a file is searched for whose name is given by gt_{dataset_name}.json. If the file is found,
        then the object given by json.load(.) is returned.
        """
        file_path = os.path.join(
            CacheManagerSingleton.GROUND_TRUTH_PATH, "gt_" + dataset_name + ".json"
        )
        if not os.path.exists(file_path):
            return None
        ret: Dict
        with open(file_path, "r") as f:
            ret = json.load(f)
        return ret

    @staticmethod
    def cache_ground_truth_data(
        gt_data: GTDataSet,
        dataset_name: str,
        corresponding_map_names: Optional[List[str]] = None,
    ) -> None:
        """Serialize the ground truth data object and save it in the ground truth directory under the name
        gt_{dataset_name}.json.

        Args:
            gt_data: Ground truth data set
            dataset_name: Name to associate with the ground truth data
            corresponding_map_names: Map names associated with the ground truth data. If the ground truth data cache
             already contains this data set and other corresponding map names, then this list extends that list.
        """
        if corresponding_map_names is None:
            corresponding_map_names = []

        if not os.path.exists(CacheManagerSingleton.GROUND_TRUTH_PATH):
            os.mkdir(CacheManagerSingleton.GROUND_TRUTH_PATH)
        gt_dict = gt_data.dict()
        file_name = "gt_" + dataset_name + ".json"
        with open(
            os.path.join(CacheManagerSingleton.GROUND_TRUTH_PATH, file_name), "w"
        ) as f:
            json.dump(gt_dict, f, indent=2)

        ground_truth_mapping_dict: Dict[str, List[str]]
        if os.path.exists(CacheManagerSingleton.GROUND_TRUTH_MAPPING_PATH):
            with open(CacheManagerSingleton.GROUND_TRUTH_MAPPING_PATH, "r") as f:
                ground_truth_mapping_dict = json.load(f)
        else:
            ground_truth_mapping_dict = dict(GROUND_TRUTH_MAPPING_STARTING_PT)

        if dataset_name in ground_truth_mapping_dict:
            ground_truth_mapping_dict[dataset_name].extend(corresponding_map_names)
        else:
            ground_truth_mapping_dict[dataset_name] = list(corresponding_map_names)
        with open(CacheManagerSingleton.GROUND_TRUTH_MAPPING_PATH, "w") as f:
            json.dump(ground_truth_mapping_dict, f, indent=2)

    @staticmethod
    def cache_sweep_results(
        sr: Union[OSweepResults, OMultiSweepResult], file_name: str
    ):
        """Serialize the provided pydantic model and write as a json file to the file specified by the file name under
        the cache subdirectory CacheManagerSingleton.SWEEP_RESULTS_PARENT.
        """
        if not file_name.endswith(".json"):
            file_name += ".json"

        if not os.path.exists(CacheManagerSingleton.SWEEP_RESULTS_PATH):
            os.mkdir(CacheManagerSingleton.SWEEP_RESULTS_PATH)

        if sr.sweep_args is not None:
            sr.sweep_args = None

        with open(
            os.path.join(CacheManagerSingleton.SWEEP_RESULTS_PATH, file_name), "w"
        ) as f:
            f.write(sr.json(indent=2))

    @staticmethod
    def cache_pgt_validation_results(
        results: OResultPseudoGTMetricValidation, file_name: str
    ):
        """Serialize the provided pydantic model and write as a json file to the file specified by the file name under
        the cache subdirectory CacheManagerSingleton.PGT_VALIDATION_RESULTS_PATH.
        """
        if not file_name.endswith(".json"):
            file_name += ".json"

        if not os.path.exists(CacheManagerSingleton.PGT_VALIDATION_RESULTS_PATH):
            os.mkdir(CacheManagerSingleton.PGT_VALIDATION_RESULTS_PATH)

        with open(
            os.path.join(CacheManagerSingleton.PGT_VALIDATION_RESULTS_PATH, file_name),
            "w",
        ) as f:
            f.write(results.json(indent=2))

    # -- Private static methods

    @staticmethod
    def _read_cache_directory(key: str) -> Union[str, None]:
        """Reads the dictionary stored as a json file in <cache folder>/directory.json and returns the value
        associated with the specified key. The key-value pairs in the directory.json map file names to map names.

        Note that no error handling is implemented.

        Args:
            key (str): Key to query the dictionary

        Returns:
            Value associated with the key
        """
        with open(
            os.path.join(CacheManagerSingleton.CACHE_PATH, "directory.json"), "r"
        ) as directory_file:
            directory_json = json.loads(directory_file.read())
            directory_file.close()
            loaded = True
        if loaded:
            return directory_json.get(key)
        else:
            return None

    @staticmethod
    def _resolve_cache_dir() -> bool:
        """Returns true if the cache folder exists, and attempts to create a new one if there is none.

        A file named directory.json is also created in the cache folder.

        This method catches all exceptions associated with creating new directories/files and displays a corresponding
        diagnostic message.

        Returns:
            True if no exceptions were caught and False otherwise
        """
        if not os.path.exists(CacheManagerSingleton.CACHE_PATH):
            try:
                os.mkdir(CacheManagerSingleton.CACHE_PATH)
            except Exception as ex:
                print(
                    f"Could not create a cache directory at {CacheManagerSingleton.CACHE_PATH} due to error: {ex}"
                )
                return False

        directory_path = os.path.join(
            CacheManagerSingleton.CACHE_PATH, "directory.json"
        )
        if not os.path.exists(directory_path):
            try:
                with open(
                    os.path.join(CacheManagerSingleton.CACHE_PATH, "directory.json"),
                    "w",
                ) as directory_file:
                    directory_file.write(json.dumps({}))
                    directory_file.close()
                return True
            except Exception as ex:
                print(
                    "Could not create {} file due to error: {}".format(
                        directory_path, ex
                    )
                )
        else:
            return True

    @staticmethod
    def _append_to_cache_directory(key: str, value: str) -> None:
        """Appends the specified key-value pair to the dictionary stored as a json file in
        <cache folder>/directory.json.

        If the key already exists in the dictionary, its value is overwritten. Note that no error handling is
        implemented.

        Args:
            key (str): Key to store value in
            value (str): Value to store under key
        """
        directory_json_path = os.path.join(
            CacheManagerSingleton.CACHE_PATH, "directory.json"
        )
        with open(directory_json_path, "r") as directory_file_read:
            directory_json = json.loads(directory_file_read.read())
            directory_file_read.close()
        directory_json[key] = value
        new_directory_json = json.dumps(directory_json, indent=2)
        with open(directory_json_path, "w") as directory_file_write:
            directory_file_write.write(new_directory_json)
            directory_file_write.close()

    @staticmethod
    def flatten_firebase(tree):
        """
        Converts Firebase Tree into List
        """
        map_list = []
        for value in tree.values():
            if isinstance(value, dict):
                map_list.extend(CacheManagerSingleton.flatten_firebase(value))
            else:
                map_list.append(value)
        return map_list

    # -- Private instance methods --

    def set_credentials(
        self, credentials: firebase_admin.credentials.Certificate
    ) -> None:
        """Instantiates a firebase app with the credentials. If the app has already been initialized, then no action is
        taken.

        Notes:
            Acquires the __synch_mutex (calling from another thread will block until this completes).

        Args:
            credentials: Firebase credentials
        """
        with self.__synch_mutex:
            self.__app = firebase_admin.initialize_app(
                credentials, self.__app_initialize_dict
            )
            self.__bucket = storage.bucket(app=self.__app)
            self.__db_ref = db.reference(f"/{self.UNPROCESSED_MAPS_PARENT}")
            self.__were_credentials_set = True

    def download_maps_for_device(self, device_id_name: str):
        """Download all maps for firebase for the specified device_id_name"""
        device_config_file = open(self.FIREBASE_CONFIG_PATH, "r")
        device_config = json.loads(device_config_file.read())
        if device_id_name not in device_config.keys():
            raise KeyError(
                "User specified device_id_name that is not in firebase_device_config"
            )
        device_id = device_config[device_id_name]
        map_info = db.reference(f"{self.UNPROCESSED_MAPS_PARENT}/{device_id}").get()
        return self._download_all_maps_recur(map_info=map_info)

    def _download_all_maps_recur(
        self, map_info: Union[Dict[str, Dict], None] = None, uid: str = None
    ):
        """Recursive function for downloading all maps from Firebase."""
        if map_info is None:
            map_info = db.reference(self.UNPROCESSED_MAPS_PARENT).get()

        for child_key, child_val in map_info.items():
            if isinstance(child_val, str):
                print(f'Downloading {"" if uid is None else uid + "/"}{child_key}')
                self._firebase_get_and_cache_unprocessed_map(
                    child_key, child_val, uid=uid
                )
            elif isinstance(child_val, dict):
                self._download_all_maps_recur(
                    map_info=child_val, uid=child_key if uid is None else uid
                )

    def _combine_maps_with_shared_cloud_anchors(
        self,
        map_info: List = None,
        map_seed: Union[str, MapInfo] = None,
    ) -> MapInfo:
        """
        Iterative function for finding maps with shared Cloud Anchors

        Args:
        map_info: A List of Firebase addresses for the location of each map
        map_seed: A MapInfo object of the map to be added to

        Return:
        map_seed: The combined map of map_seed and all maps called from map_info
        If no shared anchors are found, return back the original map
        """
        # MAP_LIMIT = 5
        seed_anchors = set()
        maps_added = 0
        intersections = []
        if isinstance(map_seed, str):
            map_seed = list(CacheManagerSingleton.find_maps(map_seed))[0]
        for cloud_data in map_seed.map_dct["cloud_data"]:
            for instance in cloud_data:
                seed_anchors.add(instance["cloudIdentifier"])
        for address in map_info:
            # if maps_added == MAP_LIMIT:
            #     break
            info = self._check_cloud_anchor_info(address)
            if info is not None:
                if len(info[0]) >= 1:
                    intersect = seed_anchors.intersection(info[0])
                    if len(intersect) >= 1:
                        map_seed = CacheManagerSingleton.combine_maps(map_seed, info[1])
                        seed_anchors.update(info[0])
                        maps_added += 1
                        print(f"Combining {info[1].map_name}...")
                        intersections.append(intersect)
        if len(intersections) == 0:
            print("No maps with shared Cloud Anchors found in database")
        print(f"{maps_added} Maps Combined")
        return map_seed

    def _check_cloud_anchor_info(self, map_json: str, uid: str = None):
        """
        Return a touple containing a list of all the Cloud Identifier ID in the map from the given Firebase address
        and the MapInfo object of the called map

        Args:
        map_json: A string of the Firebase address where the map is located
        """
        if self.__max_listen_wait > 0:
            self.__timer_mutex.acquire()
            self.__listen_kill_timer.cancel()
            self.__listen_kill_timer = Timer(
                self.__max_listen_wait, self.__firebase_listen_sem.release
            )
            self.__listen_kill_timer.start()
            self.__timer_mutex.release()

        json_blob = None
        map_name = map_json.split("/")[1]
        map_info = MapInfo(map_name, map_json, uid=uid)
        if "fixedtesting" in map_info.map_json_blob_name:
            json_blob = self.__bucket.get_blob(map_info.map_json_blob_name)
        anchors = []
        if json_blob is not None:
            json_data = json_blob.download_as_bytes()
            json_dct = json.loads(json_data)
            map_info.map_dct = json_dct
            for cloud_data in json_dct["cloud_data"]:
                for instance in cloud_data:
                    anchors.append(instance["cloudIdentifier"])
            anchors = set(anchors)
            return (anchors, map_info)
        else:
            return None

    def _firebase_get_and_cache_unprocessed_map(
        self, map_name: str, map_json: str, uid: str = None
    ) -> Union[MapInfo, None]:
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
        if self.__max_listen_wait > 0:
            self.__timer_mutex.acquire()
            self.__listen_kill_timer.cancel()
            self.__listen_kill_timer = Timer(
                self.__max_listen_wait, self.__firebase_listen_sem.release
            )
            self.__listen_kill_timer.start()
            self.__timer_mutex.release()

        map_info = MapInfo(map_name, map_json, uid=uid)
        json_blob = self.__bucket.get_blob(map_info.map_json_blob_name)
        if json_blob is not None:
            json_data = json_blob.download_as_bytes()
            json_dct = json.loads(json_data)
            map_info.map_dct = json_dct
            CacheManagerSingleton.cache_json_map(
                self.UNPROCESSED_MAPS_PARENT, map_info, json.dumps(json_dct, indent=2)
            )
            return map_info
        else:
            print("Map '{}' was missing".format(map_info.map_name))
            return None
