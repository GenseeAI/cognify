class Protection:
    def __init__(self, instance, protected_fields):
        """
        :param instance: The object whose attributes should be protected.
        :param protected_fields: List of fields or methods to protect.
        """
        self.instance = instance
        self.protected_fields = protected_fields
        self.original_attrs = {}

    def __enter__(self):
        """Temporarily overrides specified attributes with a function that raises an error."""
        for field in self.protected_fields:
            if hasattr(self.instance, field):
                # Save original attribute
                self.original_attrs[field] = getattr(self.instance, field)
                # Replace with function that raises an error
                setattr(self.instance, field, self._make_raise_error(field))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores the original attributes when the context exits."""
        for field, original_value in self.original_attrs.items():
            setattr(self.instance, field, original_value)

    def _make_raise_error(self, field):
        def raise_error(*args, **kwargs):
            raise Exception(f"Access to protected '{field}' is restricted.")

        return raise_error
