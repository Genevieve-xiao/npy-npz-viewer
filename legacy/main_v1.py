"""
NPY/NPZ 数据查看器 - 语义驱动版
程序入口
"""
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window_new import MainWindowNew


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("NPY/NPZ 数据查看器 - 语义驱动版")
    app.setOrganizationName("DataViewer")

    # 创建并显示主窗口
    window = MainWindowNew()
    window.show()

    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
