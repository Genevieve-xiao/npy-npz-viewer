"""
文件加载模块
负责加载和验证 .npy 和 .npz 文件
"""
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional


class ArrayLoader:
    """数组文件加载器"""

    def __init__(self):
        self.current_file: Optional[str] = None
        self.file_type: Optional[str] = None  # 'npy' or 'npz'
        self.npz_data: Optional[np.lib.npyio.NpzFile] = None
        self.current_array: Optional[np.ndarray] = None
        self.current_key: Optional[str] = None

    def load_file(self, file_path: str) -> Dict:
        """
        加载 .npy 或 .npz 文件

        Returns:
            dict: {
                'success': bool,
                'file_type': str,
                'keys': list (仅 npz),
                'array': np.ndarray (仅 npy),
                'error': str (如果失败)
            }
        """
        path = Path(file_path)

        # 验证文件存在
        if not path.exists():
            return {'success': False, 'error': '文件不存在'}

        # 验证文件扩展名
        suffix = path.suffix.lower()
        if suffix not in ['.npy', '.npz']:
            return {'success': False, 'error': '不支持的文件格式，仅支持 .npy 和 .npz'}

        try:
            self.close()
            if suffix == '.npy':
                return self._load_npy(file_path)
            else:
                return self._load_npz(file_path)
        except Exception as e:
            return {'success': False, 'error': f'文件加载失败: {str(e)}'}

    def _load_npy(self, file_path: str) -> Dict:
        """加载 .npy 文件"""
        try:
            array = np.load(file_path, mmap_mode='r')
            self.current_file = file_path
            self.file_type = 'npy'
            self.current_array = array
            self.current_key = None
            return {
                'success': True,
                'file_type': 'npy',
                'array': array
            }
        except Exception as e:
            return {'success': False, 'error': f'NPY 文件损坏或格式错误: {str(e)}'}

    def _load_npz(self, file_path: str) -> Dict:
        """加载 .npz 文件"""
        try:
            npz = np.load(file_path)
            keys = list(npz.keys())
            if len(keys) == 0:
                return {'success': False, 'error': 'NPZ 文件为空，没有数组'}

            self.current_file = file_path
            self.file_type = 'npz'
            self.npz_data = npz
            self.current_key = keys[0]
            self.current_array = npz[keys[0]]

            return {
                'success': True,
                'file_type': 'npz',
                'keys': keys
            }
        except Exception as e:
            return {'success': False, 'error': f'NPZ 文件损坏或格式错误: {str(e)}'}

    def switch_npz_key(self, key: str) -> Dict:
        """切换 npz 文件中的数组"""
        if self.file_type != 'npz' or self.npz_data is None:
            return {'success': False, 'error': '当前未加载 NPZ 文件'}

        if key not in self.npz_data.keys():
            return {'success': False, 'error': f'键 "{key}" 不存在'}

        try:
            self.current_key = key
            self.current_array = self.npz_data[key]
            return {'success': True, 'array': self.current_array}
        except Exception as e:
            return {'success': False, 'error': f'切换数组失败: {str(e)}'}

    def get_current_array(self) -> Optional[np.ndarray]:
        """获取当前数组"""
        return self.current_array

    def close(self):
        """关闭文件"""
        if self.npz_data is not None:
            self.npz_data.close()
        self.current_file = None
        self.file_type = None
        self.npz_data = None
        self.current_array = None
        self.current_key = None
