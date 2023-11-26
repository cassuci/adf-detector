from dataset_reader.RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)


# --------------RawBoost data augmentation algorithms---------------------------##


def process_Rawboost_feature(feature, sr, args, algo):
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:
        feature = feature

    return feature
