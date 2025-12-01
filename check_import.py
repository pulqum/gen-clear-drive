try:
    from ultralytics.utils.metrics import ap_per_class
    import torch
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
