# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np
import h5py
import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        
        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)
            
        if img.ndim == 2:
            img = img[..., np.newaxis]
            
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadH5Data(BaseTransform):
    """Load complex data from H5 file and split into real/imaginary parts.

    Required Keys:

    - h5_path (should be the path to a H5 file containing a 512x512 complex array)

    Modified Keys:

    - img (will be the 512x512x2 array, [real, imaginary])
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded array to float32.
            Defaults to True (common for deep learning inputs).
        dataset_key (str): The key name to access the complex dataset in the H5 file.
            Defaults to 'dataset'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty array or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. Defaults to None. New in version 2.0.0rc4.
    """

    def __init__(self,
                 magnitude : bool = False,
                 power_dB : bool = False,
                 power_dB_norm : bool = False,
                 power_dB_no_norm_mean_std : bool = False,
                 power : bool = False,
                 dB_t75_power : bool = False,
                 magnitude_phase:bool = False,
                 IQ_abs: bool = False,
                 complex: bool = False,
                 to_float32: bool = False,  # 默认不转换为float32
                 dataset_key: str = 'dataset',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:

        self.ignore_empty = ignore_empty
        self.magnitude = magnitude
        self.power_dB = power_dB
        self.power_dB_norm = power_dB_norm
        self.power_dB_no_norm_mean_std = power_dB_no_norm_mean_std
        self.magnitude_phase = magnitude_phase
        self.power = power
        self.dB_t75_power = dB_t75_power
        self.IQ_abs = IQ_abs
        self.to_float32 = to_float32
        self.dataset_key = dataset_key  # 存储复数数据的数据集键名
        self.complex = complex

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None

        # 参数互斥校验
        if self.complex and (self.IQ_abs or self.IQ_abs_log):
            raise ValueError("'complex' cannot be True with 'IQ_abs' or 'IQ_abs_log'")
        if self.IQ_abs and self.IQ_abs_log:
            raise ValueError("'IQ_abs' and 'IQ_abs_log' cannot be True at the same time")

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')
            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def normalize_and_combine_channels(self, I_channel, Q_channel, num_stages=3, decay_factor=0.1):
        """
        对给定的I和Q通道数据分别进行分阶段最大最小归一化，然后将结果拼接成一个3D数组。

        Args:
            I_channel (np.ndarray): 处理好的I通道数据，2D数组 (height, width)。
            Q_channel (np.ndarray): 处理好的Q通道数据，2D数组 (height, width)。
            num_stages (int, optional): 每个通道的归一化阶段数。默认值为3。
            decay_factor (float, optional): 每次归一化最大值的衰减因子。默认值为0.1。

        Returns:
            np.ndarray: 拼接后的3D数组，形状为 (6, height, width)，
                       顺序为 [I1, I2, I3, Q1, Q2, Q3]。
        """
        all_normalized = []

        # 处理I通道
        current_max = np.max(I_channel)
        current_min = np.min(I_channel)
        denominator = current_max - current_min

        # 避免除零错误
        if denominator < 1e-12:
            denominator = 1e-12

        for _ in range(num_stages):
            normalized = (I_channel - current_min) / denominator
            all_normalized.append(np.clip(normalized, 0, 1))
            current_max *= decay_factor
            denominator = current_max - current_min  # 更新分母

        # 处理Q通道
        current_max = np.max(Q_channel)
        current_min = np.min(Q_channel)
        denominator = current_max - current_min

        if denominator < 1e-12:
            denominator = 1e-12

        for _ in range(num_stages):
            normalized = (Q_channel - current_min) / denominator
            all_normalized.append(np.clip(normalized, 0, 1))
            current_max *= decay_factor
            denominator = current_max - current_min  # 更新分母

        # 拼接所有通道并返回
        return np.stack(all_normalized, axis=-1)
        
    def transform(self, results: dict) -> Optional[dict]:
        """Load complex data from H5 and split into real/imaginary parts.

        Args:
            results (dict): Result dict with 'h5_path' key.

        Returns:
            dict: Updated with 'img', 'img_shape', 'ori_shape' keys.
        """
        filename = results['img_path']
        try:
            # 通过 file_client 读取文件字节流
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                file_bytes = file_client.get(filename)  # 读取字节流
                h5_file = h5py.File(BytesIO(file_bytes), 'r')  # 从内存字节流解析
            else:
                h5_file = h5py.File(filename, 'r')  # 本地文件直接打开

            with h5_file:  # 确保文件正确关闭
                complex_data = h5_file[self.dataset_key][()]

            # 验证复数数组形状
            if complex_data.ndim != 2 or complex_data.shape != (512, 512):
                raise ValueError(f"Expected complex array shape (512, 512), but got {complex_data.shape}")
            # 读取出来的时候应该是512,512的复数

            if self.complex:
                img = np.array(complex_data, dtype=np.complex64)
            else:
                if self.magnitude:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part**2+imag_part**2)
                    
                elif self.power:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    power = real_part ** 2 + imag_part ** 2

                elif self.power_dB:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part**2+imag_part**2)
                    power_dB = 20 * np.log10(magnitude)
                                   
                elif self.power_dB_norm:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part**2+imag_part**2)
                    epsilon = 1e-5
                    magnitude_safe = np.maximum(magnitude, epsilon)  # >=epsilon
                    power_dB = 20 * np.log10(magnitude_safe)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB-power_dB_min)/(power_dB_max-power_dB_min)
                   
                elif self.power_dB_no_norm_mean_std:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    power= real_part ** 2 + imag_part ** 2
                    power_dB = 10 * np.log10(power)
                    mean = -30.084263589092878
                    std = 11.404058539922564
                    norm_power_dB = (power_dB - mean ) / std

                elif self.magnitude_phase:
                    real_part = complex_data.real
                    imag_part = complex_data.imag
                    magnitude = np.sqrt(real_part**2+imag_part**2)
                    theta = np.arctan2(imag_part, real_part)
                    phase = np.mod(theta, 2*np.pi)
                    
                elif self.dB_t75_power:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
                    power = magnitude ** 2
                    
                    power_dB = 20 * np.log10(magnitude)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min) * 255
                    
                    t75 = np.percentile(norm_power_dB,75)
                    dB_75 = np.where(norm_power_dB>t75, norm_power_dB,0)
                    power_norm = (power - np.min(power))/(np.max(power)-np.min(power)) *255

                    
                elif self.IQ_abs:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    
                else:
                    real_part = complex_data.real
                    imag_part = complex_data.imag
                    img = np.stack([real_part, imag_part], axis=-1)  # 形状：(512, 512, 2)


                if self.IQ_abs:
	    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    img = np.stack([real_part, imag_part], axis=-1)  # 形状：(512, 512, 2)

                if self.magnitude:
                    img  = magnitude
                
                if self.power_dB:
                    img = power_dB
                    
                if self.power_dB_norm:
                    img = norm_power_dB

                if self.power_dB_no_norm_mean_std :
                    img = norm_power_dB

                if self.dB_t75_power:
                    img = np.stack([norm_power_dB, dB_75, power_norm], axis=-1)

                if self.power:
                    img = power

                if self.magnitude_phase:
                    img = np.stack([magnitude, phase], axis=-1)
                    
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        
        if self.to_float32:
            img = img.astype(np.float32)
        # 这一步主要是当输入的数据是复数的时候，只有2维h,w.所以维度不够，需要扩展维度
        if img.ndim == 2:
            img = img[..., np.newaxis]

        results['img'] = img
        results['img_shape'] = img.shape[:2]  # (512, 512)
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"dataset_key='{self.dataset_key}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(BaseTransform):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in key point detection.
                # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
                # means the visibility of this keypoint. n must be equal to the
                # number of keypoint categories.
                'keypoints': [x1, y1, v1, ..., xn, yn, vn]
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4)
             # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # with (x, y, v) order, in np.float32 type.
            'gt_keypoints': np.ndarray(N, NK, 3)
        }

    Required Keys:

    - instances

      - bbox (optional)
      - bbox_label
      - keypoints (optional)

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_seg_map (np.uint8)
    - gt_keypoints (np.float32)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Defaults to False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_keypoints: bool = False,
        imdecode_backend: str = 'cv2',
        file_client_args: Optional[dict] = None,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                self.file_client_args, results['seg_map_path'])
            img_bytes = file_client.get(results['seg_map_path'])
        else:
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)

        results['gt_seg_map'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
