import inspect
import re
import typing
from abc import ABC, abstractmethod
from json import JSONEncoder
from typing import Dict, Any, Tuple, Iterable, List, Optional, Set, Union


class CouldNotCheckTypeException(Exception):
    def __init__(self, message):
        self.message = message


class InvalidInstanceVariableType(Exception):
    def __init__(self, message):
        self.message = message


class SubscriptedTypeChecker(ABC):
    """Enables type checking with type specifications given by typing._GenericAlias types, including subscripted List
    and Tuple generic aliases (which would otherwise raise an exception if checked by Python normally at runtime).

    This is an abstract class and is intended to provide capabilities solely to its child classes.
    """
    __primitives_allowed = {"float", "int", "str", "None"}
    __accepted_subscripted_types = {
        "typing.List": "list",
        "typing.Tuple": "tuple"
    }
    class_attr_dict: Dict[str, Any]

    @abstractmethod
    def __init__(self, instance_attr_strs_to_check: Iterable[str]):
        """
        Args:
            instance_attr_strs_to_check: An iterable object containing the strings specifying the names of this
             instance's attributes to check the types of.

        Raises:
             CouldNotCheckTypeException: An error occurred when trying to verify an object's type.
             InvalidInstanceVariableType: One of the instance attributes was incorrect.
        """
        all_class_attrs = inspect.getmembers(self.__class__, lambda a: not (inspect.isroutine(a)))
        self.__class__.class_attr_dict = {attr[0]: attr[1] for attr in all_class_attrs if attr[0].startswith("type_")}
        self.check_instance_attrs_match_expected_types_and_raise(instance_attr_strs_to_check)

    def check_instance_attrs_match_expected_types_and_raise(self, instance_attr_strs: Iterable[str]) -> None:
        """Based on the results of invoking check_instance_attrs_match_expected_types, raises an exception if any
        of the instance attributes checked are not of the correct type.

        Args:
            instance_attr_strs: Names of instance attributes to check.

        Raises:
             InvalidInstanceVariableType: if one or more of the instance attributes is not of the correct type.
        """
        results = self.check_instance_attrs_match_expected_types(instance_attr_strs)
        if not results[0]:
            raise InvalidInstanceVariableType(
                f"The following variables are of the incorrect type as indicated by their corresponding class "
                f"attributes: {[item[0] for item in results[1].items() if not item[1]]}"
            )

    def check_instance_attrs_match_expected_types(self, instance_attr_strs: Iterable[str]) \
            -> Tuple[bool, Dict[str, bool]]:
        """For each string in instance_attr_strs, invokes the __check_type method with the corresponding instance
        attribute. See that method for more details.

        Args:
            instance_attr_strs: Names of instance attributes to check.

        Returns:
            A tuple whose second value is a dictionary mapping instance attribute names to True if they are of the
             correct type and False otherwise. The first value in the tuple is True only if all values in the dictionary
             are true.
        """
        all_correct = True
        results = {}
        for instance_attr_str in instance_attr_strs:
            res = self.__check_type(self.__getattribute__(instance_attr_str), instance_attr_str)
            if not res:
                all_correct = False
                results[instance_attr_str] = False
            else:
                results[instance_attr_str] = True
        return all_correct, results

    @classmethod
    def __check_type(cls, o: Any, attr_name: str) -> bool:
        """Checks whether the provided object is of the correct type as specified by the corresponding class attribute.

        The corresponding class attribute's name is expected to be given by "type_" + attr_name. The value of said class
        attribute is expected to be of type `typing.Type`. This class's __primitives_allowed and
        __accepted_subscripted_types attributes give the strings of the allowable types for said class attribute.

        Args:
            o: Object whose type is checked.
            attr_name: Class attribute's name to reference. Should be the name of the "o" argument with the prefix \
             "type_". List types should only specify one inner type, but Tuple types can specify any number of inner
             types.

        Returns:
            True if o is of the type indicated by the corresponding attribute.
        """
        expected_type: Any
        try:
            expected_type = cls.class_attr_dict["type_" + attr_name]
        except KeyError:
            raise CouldNotCheckTypeException(
                f"Could not find expected class attribute type_{attr_name}; did you make sure that there is a class "
                f"attribute for every instance attribute that signifies the expected type of the instance attribute "
                f"and is named the same?")

        # noinspection PyProtectedMember
        if isinstance(expected_type, typing._GenericAlias):
            expected_type_str = str(expected_type)
            found = re.findall(r"(\[|]|[\w.]+)", expected_type_str)
            for type_str_idx, type_str in enumerate(found):
                if type_str in SubscriptedTypeChecker.__accepted_subscripted_types:
                    found[type_str_idx] = SubscriptedTypeChecker.__accepted_subscripted_types[type_str]
            return SubscriptedTypeChecker.__obj_matches_type_str(o, found)
        elif isinstance(expected_type, type):
            this_obj_type_str = type(o).__name__
            # Handle special cases of floating points to allow equivalence between Numpy's float64 and float32 types
            # with the generic float type (TODO: better way to handle this?)
            if this_obj_type_str == "float64" or this_obj_type_str == "float32":
                this_obj_type_str = "float"
            return this_obj_type_str == expected_type.__name__
        else:
            raise CouldNotCheckTypeException(
                f"Expected type of attribute is of type {type(expected_type).__name__}, which cannot be handled"
            )

    @staticmethod
    def __obj_matches_type_str(o, type_str_list: List[str], _idx=0) -> bool:
        """Checks whether an object matches the type specified by the list of strings. Implemented recursively.

        Examples:
            - SubscriptedTypeChecker.__obj_matches_type_str(0.0, "float")  # Evaluates to True
            - SubscriptedTypeChecker.__obj_matches_type_str([1, 2, 3], ["list", "[", "int", "]"])  # Evaluates to True
            - SubscriptedTypeChecker.__obj_matches_type_str([1.1, 2.2], ["list", "[", "int", "]"])  # Evaluates to False
            - SubscriptedTypeChecker.__obj_matches_type_str((1.1, 2.2), ["tuple", "[", "float", "float", "]"])
              # Evaluates to True
            - SubscriptedTypeChecker.__obj_matches_type_str((1.1), ["tuple", "[", "float", "float", "]"])
              # Evaluates to False (expected two floats, only got one)
        Args:
            o: Object whose type is checked
            type_str_list: List of types as strings. Subscriptable types can optionally a pair of square brackets and
             that enclose type string(s) indicating the types of elements that can be contained. Supported subscriptable
             types are given in SubscriptedTypeChecker.__accepted_subscripted_types. List types should only specify one
             inner type, but Tuple types can specify any number of inner types.
            _idx: Index of the type_str_list argument (to be used in recursive call).

        Returns:
            True if the object matches the specified type, False otherwise.
        """
        this_expected_type_str = type_str_list[_idx]
        this_obj_type_str = type(o).__name__

        # Handle special cases of floating points to allow equivalence between Numpy's float64 and float32 types with
        # the generic float type (TODO: better way to handle this?)
        if this_obj_type_str == "float64" or this_obj_type_str == "float32":
            this_obj_type_str = "float"

        if this_expected_type_str in SubscriptedTypeChecker.__primitives_allowed:
            return this_expected_type_str == this_obj_type_str
        elif this_obj_type_str in SubscriptedTypeChecker.__primitives_allowed:
            return False  # o was found to be a primitive type but another type was expected
        elif this_expected_type_str in SubscriptedTypeChecker.__accepted_subscripted_types.values():
            # Expecting that o is a list or tuple
            if this_obj_type_str != this_expected_type_str:
                return False

            # If there is no specified contained type of the subscripted type, then return True. For following
            # conditional statement, rely on short-circuiting to avoid index error.
            if len(type_str_list) - 1 == _idx or type_str_list[_idx + 1] != "[":
                return True

            if this_expected_type_str == "tuple":
                # If is a tuple of specified length, check that it matches the expected length
                expected_tuple_len = 0
                _idx += 2  # Skip over following entry, which is an open bracket
                tuple_start_idx = _idx
                while type_str_list[_idx] != "]":
                    _idx += 1
                    expected_tuple_len += 1
                if len(o) != expected_tuple_len:
                    return False

                # Now recurse on each entry of the tuple
                _idx = tuple_start_idx
                o_idx = 0
                while o_idx < expected_tuple_len:
                    if not SubscriptedTypeChecker.__obj_matches_type_str(o[o_idx], type_str_list, _idx):
                        return False
                    o_idx += 1
                    _idx += 1

                # If this point is reached, then each element of the tuple matched.
                _idx += 2  # Skip over closed bracket
                return True
            else:  # Can only be a list if not a tuple
                # If is a list, then check that its inner type is correct
                _idx += 2  # Skip over open bracket
                for o_element in o:
                    if not SubscriptedTypeChecker.__obj_matches_type_str(o_element, type_str_list, _idx):
                        return False

                # The List elements checked out, so return True
                _idx += 2  # Skip over closed bracket
                return True
        elif issubclass(o.__class__, SubscriptedTypeChecker):
            return True  # Assumes that check has been run already because the object was successfully constructed
        else:
            raise CouldNotCheckTypeException(f"Encountered unhandled expected type {this_expected_type_str}. ")


class TypeCheckingJSONEncoder(JSONEncoder, SubscriptedTypeChecker, ABC):
    """Enables serialization and automatic type checking of any child classes' instance attributes.

    Notes:
        For every instance value, it is expected that there is a corresponding class attribute given by the same name
        (except prefixed with "type_") that prescribes the expected type of the instance attribute. Any instance
        attributes that are to be excluded from this type checking should be indicated in the attrs_to_skip_checking_of
        argument of this class's constructor.

        For type checking to correctly occur, this class's constructor must be called *after* the instance values are
        instantiated.

    Attributes:
        attr_strs_to_serialize: Set of strings of attributes that are
    """
    attrs_to_skip = set(JSONEncoder().__dict__.keys())

    @abstractmethod
    def __init__(self, attrs_to_skip_checking_of: Optional[Set[str]] = None,
                 attrs_to_set: Optional[Dict[str, object]] = None):
        """Abstract initializer intended to be extended.

        If invoked at the end of the child class's initialization, checks the type-correctness of all the child
        class's instance attributes (except those specified by the attrs_to_skip_checking_of argument).

        Args:
            attrs_to_skip_checking_of: A set of strings specifying instance attributes to exclude from the type checking
            attrs_to_set: A dictionary whose key-value pairs are used to set attributes' names and their contents for
             this instance being initialized.

        Raises:
            InvalidInstanceVariableType if there is a discrepancy in types. Refer to SubscriptedTypeChecker's
             initializer documentation for more information.
        """
        if attrs_to_set is not None:
            for item in attrs_to_set.items():
                self.__setattr__(item[0], item[1])

        self.attr_strs_to_serialize: Set[str] = set(self.__dict__.keys()) \
            .difference(TypeCheckingJSONEncoder.attrs_to_skip,
                        attrs_to_skip_checking_of if attrs_to_skip_checking_of is not None else {})
        super().__init__()
        SubscriptedTypeChecker.__init__(self, instance_attr_strs_to_check=self.attr_strs_to_serialize)

    def default(self, o: "TypeCheckingJSONEncoder") -> Dict:
        """Overrides JSONEncoder's default instance method. Converts an instance of this class into a dictionary.

        Notes:
            This method is mutually recursive with _recursive_list_builder.

        Args:
            o: Instance of this class.

        Returns:
            Dictionary representation of the object

        Raises:
            InvalidInstanceVariableType: If the specified instance attributes are not of the correct type, then an
             exception raised by check_instance_attrs_match_expected_types_and_raise will go uncaught.
        """
        super().check_instance_attrs_match_expected_types_and_raise(self.attr_strs_to_serialize)
        serialized = {}
        for attr_str in o.attr_strs_to_serialize:
            attr = o.__getattribute__(attr_str)
            if issubclass(attr.__class__, TypeCheckingJSONEncoder):
                serialized[attr_str] = attr.encode(attr)
            elif isinstance(attr, list) or isinstance(attr, tuple):
                list_to_add = self._recursive_list_builder(attr)
                serialized[attr_str] = list_to_add
            else:
                serialized[attr_str] = attr
        return serialized

    @staticmethod
    def _recursive_list_builder(o: Union[List, Tuple, "TypeCheckingJSONEncoder"]) -> \
            Union[List, Dict, "TypeCheckingJSONEncoder"]:
        if isinstance(o, list) or isinstance(o, tuple):
            return [TypeCheckingJSONEncoder._recursive_list_builder(elem) for elem in o]
        elif issubclass(o.__class__, TypeCheckingJSONEncoder):
            return o.default(o)
        else:
            return o


if __name__ == "__main__":
    # Following is an example that does not produce any errors. You will notice that an exception is raised in one of
    # two cases:
    # 1. You change the instance attribute's type such that it no longer matches the type specified by the
    #    corresponding class attribute.
    # 2. (and/or) You create a scenario that is not capable of being successfully checked (e.g., type checking of
    #    dictionaries is not currently supported).

    class ComposedExample(TypeCheckingJSONEncoder):
        type_test_4 = List

        def __init__(self):
            self.test_4 = [1, 2, 3]
            super().__init__()


    class Example(TypeCheckingJSONEncoder):
        type_test_1 = List[int]
        type_test_2 = List[List]
        type_test_3 = Tuple[float, float, float]
        type_test_recur = Tuple[ComposedExample, ComposedExample]

        def __init__(self):
            self.test_1 = []
            self.test_2 = [[1, 2], ["a", "b"]]
            self.test_3 = (0.2, 0.2, 0.2)

            # Test composition of other subclasses of TypeCheckingJSONEncoder
            self.test_recur = (ComposedExample(), ComposedExample())

            self.exception = "exception"  # Skip type checks on this
            super().__init__(attrs_to_skip_checking_of={"exception"})


    ex = Example()
    print(ex.default(ex))  # Print serializable representation
