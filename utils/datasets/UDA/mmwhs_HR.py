from pathlib import Path
from typing import Tuple, List

from datasets.dataloader_interface import MedicalDatasetInterface
from datasets.dataset_abstract import MedicalImageSegmentationDataset
from datasets.folderOperation_tool import downloading
from general_utils.path_tool import DATA_PATH


class CTHRDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_HR.zip"
    folder_name = "MMWHS_HR"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=f'{str(path)}/CT', mode=mode, sub_folders=sub_folders, patient_pattern=patient_pattern)

class CTHRInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            CTHRDataset,
            root_dir,
            verbose,
        )

    def _create_datasets(
            self,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            patient_pattern=r"ct_train_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            patient_pattern=r"ct_train_\d+"
        )
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            patient_pattern=r"ct_train_\d+"
        )
        return train_set, val_set, test_set

class MRHRDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "MMWHS_HR.zip"
    folder_name = "MMWHS_HR"

    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=f'{str(path)}/MR', mode=mode, sub_folders=sub_folders, patient_pattern=patient_pattern)

class MRHRInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            MRHRDataset,
            root_dir,
            verbose,
        )

    def _create_datasets(
            self,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            patient_pattern=r"mr_train_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            patient_pattern=r"mr_train_\d+"
        )
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            patient_pattern=r"mr_train_\d+"
        )
        return train_set, val_set, test_set