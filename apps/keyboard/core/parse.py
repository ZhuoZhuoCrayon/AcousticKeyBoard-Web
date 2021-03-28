# -*- coding: utf-8 -*-
import logging
from typing import List

from django.db import transaction

from apps.keyboard import constants, models
from apps.keyboard.core import format
from apps.keyboard.utils import decorators
from djangocli.constants import LogModule
from in_python.core.pre.load import DataSetUnit

logger = logging.getLogger(LogModule.APPS)


class DatasetParse:
    def __init__(self, dataset_unit: DataSetUnit, **more_info):
        self.dataset_unit = dataset_unit
        self.more_info = more_info

    @decorators.class_member_cache(name="dataset_db_obj")
    def get_dataset_db_obj(self) -> models.Dataset:
        return models.Dataset(
            dataset_name=self.dataset_unit.dataset_name,
            data_type=self.dataset_unit.data_type,
            project_type=self.dataset_unit.project_type,
            description=self.dataset_unit.description,
            length=self.dataset_unit.length,
            fs=self.dataset_unit.fs,
            verbose_name=self.more_info.get("verbose_name", self.dataset_unit.dataset_name),
            description_more=self.more_info.get("description_more", self.dataset_unit.description),
        )

    @decorators.class_member_cache(name="original_data_db_objs")
    def get_original_data_db_objs(self, dataset_id: int) -> List[models.DatasetOriginalData]:
        dataset_original_data_db_objs = []

        def _bulk_gen_db_objs(label_type: str):
            _label_type_data_map = {
                constants.LabelType.TRAIN: self.dataset_unit.train_data,
                constants.LabelType.TEST: self.dataset_unit.test_data,
            }
            for _label, _label_data in _label_type_data_map[label_type].items():
                for _vec in _label_data:
                    dataset_original_data_db_objs.append(
                        models.DatasetOriginalData(
                            dataset_id=dataset_id,
                            label=_label,
                            label_type=constants.LabelType.TRAIN,
                            original_data=_vec.tolist(),
                        )
                    )

        _bulk_gen_db_objs(label_type=constants.LabelType.TRAIN)
        _bulk_gen_db_objs(label_type=constants.LabelType.TEST)

        return dataset_original_data_db_objs

    @decorators.class_member_cache(name="mfcc_feature_db_objs")
    def get_mfcc_feature_db_objs(self, dataset_id) -> List[models.DatasetMfccFeature]:
        dataset_mfcc_feature_db_objs = []

        def _bulk_gen_db_objs(label_type: str):
            _label_type_data_map = {
                constants.LabelType.TRAIN: self.dataset_unit.train_data,
                constants.LabelType.TEST: self.dataset_unit.test_data,
            }
            feature_data = format.batch_gen_mfcc_feature(data=_label_type_data_map[label_type], fs=self.dataset_unit.fs)

            for _label, _label_data in feature_data.items():
                for _mfcc_feature in _label_data:
                    dataset_mfcc_feature_db_objs.append(
                        models.DatasetMfccFeature(
                            dataset_id=dataset_id,
                            label=_label,
                            label_type=constants.LabelType.TRAIN,
                            mfcc_feature=_mfcc_feature.tolist(),
                        )
                    )

        _bulk_gen_db_objs(label_type=constants.LabelType.TRAIN)
        _bulk_gen_db_objs(label_type=constants.LabelType.TEST)

        return dataset_mfcc_feature_db_objs

    def import_db(self, save_original_data=True) -> int:
        with transaction.atomic():
            dataset_obj = self.get_dataset_db_obj()
            dataset_obj.save()
            logger.info(
                f"created dataset: id -> {dataset_obj.id}, name -> {dataset_obj.dataset_name}, "
                f"verbose_name -> {dataset_obj.verbose_name}"
            )
            mfcc_feature_db_objs = models.DatasetMfccFeature.objects.bulk_create(
                self.get_mfcc_feature_db_objs(dataset_id=dataset_obj.id)
            )
            logger.info(f"created mfcc features: dataset_id -> {dataset_obj.id}, num -> {len(mfcc_feature_db_objs)}")

            if not save_original_data:
                logger.info("save_original_data=False, create original data skipped")
                return dataset_obj.id

            original_data_db_objs = models.DatasetOriginalData.objects.bulk_create(
                self.get_original_data_db_objs(dataset_id=dataset_obj.id)
            )
            logger.info(f"created original data: dataset_id -> {dataset_obj.id}, num -> {len(original_data_db_objs)}")

        return dataset_obj.id
