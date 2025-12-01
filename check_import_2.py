try:
    from ultralytics.utils.metrics import match_predictions
    print("Import match_predictions successful")
except ImportError:
    print("Import match_predictions failed")
