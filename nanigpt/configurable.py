from dataclasses import dataclass
from typing import ClassVar


class Configurable:
    """Base class for components that are constructed from configuration.

    Subclasses define a nested Config dataclass. The Config's build() method
    constructs the owning component, so the config tree mirrors the object tree.

    Usage:
        class MyComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                learning_rate: float = 3e-4

            def __init__(self, config: Config):
                self.lr = config.learning_rate

        config = MyComponent.Config(learning_rate=1e-4)
        component = config.build()  # returns MyComponent(config)
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        _owner: ClassVar[type | None] = None

        def build(self, **kwargs):
            """Construct the owning component from this config."""
            if self._owner is None:
                raise TypeError(
                    f"{type(self).__name__} has no owner class — "
                    f"it must be defined as a nested class inside a Configurable subclass"
                )
            return self._owner(config=self, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if not hasattr(config_cls, "__slots__"):
                raise TypeError(
                    f"{cls.__name__}.Config must use @dataclass(kw_only=True, slots=True)"
                )
            config_cls._owner = cls
