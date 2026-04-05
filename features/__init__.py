from .combined import extract_features as combined_features, feature_names as combined_names
from .mfcc_only import extract_features as mfcc_features, feature_names as mfcc_names
from .zcr_others import extract_features as zcr_features, feature_names as zcr_names

FEATURE_MODULES = {
    "combined": ("features.combined", combined_features, combined_names),
    "mfcc_only": ("features.mfcc_only", mfcc_features, mfcc_names),
    "zcr_others": ("features.zcr_others", zcr_features, zcr_names),
}
