import logging


class ModelRegistry:
    """
    Singleton registry to manage auto-labeling model classes.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class with a specific type name.
        Example:
            @ModelRegistry.register("yolov8")
            class YOLOv8(Model): ...
        """

        def decorator(model_class):
            if name in cls._registry:
                logging.warning(
                    f"Model type '{name}' is already registered. Overwriting."
                )
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by type name."""
        return cls._registry.get(name)

    @classmethod
    def list_models(cls):
        """List all registered model types."""
        return list(cls._registry.keys())
