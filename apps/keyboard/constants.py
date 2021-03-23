# -*- coding: utf-8 -*-
from typing import Tuple

from django.utils.translation import ugettext_lazy as _


class ProjectType:
    SINGLE = "single"
    COMBINE = "combine"

    @staticmethod
    def get_choices() -> Tuple:
        return (
            (ProjectType.SINGLE, _("单键")),
            (ProjectType.SINGLE, _("组合键")),
        )


class DataType:
    ALL_1MICRO = "all-1micro"
    ALL_0MICRO = "all-0micro"
    ALL_1MICRO_FAR12 = "all-1micro-far12"
    ALL_0MICRO_FAR12 = "all-0micro-far12"
    ALL_1MICRO_NEAR12 = "all-1micro-near12"
    ALL_0MICRO_NEAR12 = "all-0micro-near12"

    @staticmethod
    def get_choices() -> Tuple:
        return (
            (DataType.ALL_1MICRO, _("1号麦克风-30个键位")),
            (DataType.ALL_0MICRO, _("0号麦克风-30个键位")),
            (DataType.ALL_1MICRO_FAR12, _("1号麦克风-远间隔12个键位")),
            (DataType.ALL_0MICRO_FAR12, _("0号麦克风-远间隔12个键位")),
            (DataType.ALL_1MICRO_NEAR12, _("1号麦克风-近间隔12个键位")),
            (DataType.ALL_0MICRO_NEAR12, _("0号麦克风-近间隔12个键位")),
        )
