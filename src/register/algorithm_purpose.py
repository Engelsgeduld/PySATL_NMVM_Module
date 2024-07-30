import enum


class AlgorithmPurpose(enum.Enum):
    DEFAULT = "Any"
    NM_PARAMETRIC = "Normal Mean Parametric"
    NV_PARAMETRIC = "Normal Variance Parametric"
    NMV_PARAMETRIC = "Normal Mean-Variance Parametric"
    NM_SEMIPARAMETRIC = "Normal Mean Semiparametric"
    NV_SEMIPARAMETRIC = "Normal Variance Semiparametric"
    NMV_SEMIPARAMETRIC = "Normal Mean-Variance Semiparametric"
