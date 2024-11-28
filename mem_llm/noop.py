class Noop:
    def __getattr__(self, name):
        # Return self for any attribute access
        return self

    def __setattr__(self, name, value):
        # Ignore any attribute setting
        pass

    def __delattr__(self, name):
        # Ignore attribute deletion
        pass

    def __getitem__(self, key):
        # Ignore item access and return self
        return self

    def __setitem__(self, key, value):
        # Ignore item assignment
        pass

    def __delitem__(self, key):
        # Ignore item deletion
        pass

    def __call__(self, *args, **kwargs):
        # Ignore calls
        return self


if __name__ == "__main__":
    # Example usage
    dn = Noop()
    dn.append(10)  # Does nothing
    dn[10]  # Does nothing
    dn.some_attribute = "value"  # Does nothing
    del dn.some_attribute  # Does nothing
    dn.some_method()  # Does nothing
    print(dn)  # <DoNothing instance>
