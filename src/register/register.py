from typing import Callable, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Register of objects. Implementation of Register pattern"""

    def __init__(self, default: Optional[Type[T]] = None) -> None:
        """

        Args:
            default: Default object to return

        """
        self.default = default
        self.register_of_names: dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable:
        """Register new object

        Args:
            name: Name of lass

        Returns: Decorator function

        Raises:
            ValueError: When class with this name was registered before

        """

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self.register_of_names:
                raise ValueError("This name is already registered")
            self.register_of_names[name] = cls
            return cls

        return decorator

    def dispatch(self, name: str) -> Type[T]:
        """Find object by name

        Args:
            name: Name of class

        Returns: object

        Raises:
            ValueError: When object with this name was not found

        """
        if name in self.register_of_names:
            return self.register_of_names[name]
        if self.default is None:
            raise ValueError(f"{name} realisation not registered")
        return self.default
