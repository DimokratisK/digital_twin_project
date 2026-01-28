import importlib
module_name, class_name = "twin_core.utils.UNET_model.UNet".rsplit(".", 1)
mod = importlib.import_module(module_name)
cls = getattr(mod, class_name)
print("Imported", cls, "from", mod.__file__)
