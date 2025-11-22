from pathlib import Path
from typing import Type

from app.ocr.base import BaseOCREngine


class OCREngineRegistry:
    """
    Registry for OCR engine implementations.

    This class manages the registration and discovery of OCR engines.
    Engines are registered by name and can be instantiated with model paths.
    """

    _engines: dict[str, Type[BaseOCREngine]] = {}

    @classmethod
    def register(cls, engine_class: Type[BaseOCREngine]) -> Type[BaseOCREngine]:
        """
        Register an OCR engine class.

        Can be used as a decorator:
            @OCREngineRegistry.register
            class MyOCREngine(BaseOCREngine):
                ...

        Args:
            engine_class: The OCR engine class to register.

        Returns:
            The same engine class (for decorator usage).
        """
        # Create a temporary instance to get the name
        # We use a dummy path since we just need the name property
        temp_instance = object.__new__(engine_class)
        temp_instance.model_path = Path(".")
        temp_instance._loaded = False

        name = temp_instance.name
        cls._engines[name] = engine_class
        return engine_class

    @classmethod
    def get_engine_class(cls, name: str) -> Type[BaseOCREngine] | None:
        """
        Get an engine class by name.

        Args:
            name: The engine identifier.

        Returns:
            The engine class, or None if not found.
        """
        return cls._engines.get(name)

    @classmethod
    def create_engine(cls, name: str, model_path: Path) -> BaseOCREngine | None:
        """
        Create an instance of an engine.

        Args:
            name: The engine identifier.
            model_path: Path to the model files.

        Returns:
            An engine instance, or None if the engine type is not registered.
        """
        engine_class = cls.get_engine_class(name)
        if engine_class is None:
            return None
        return engine_class(model_path)

    @classmethod
    def list_registered(cls) -> list[str]:
        """
        List all registered engine names.

        Returns:
            List of registered engine identifiers.
        """
        return list(cls._engines.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an engine is registered."""
        return name in cls._engines
