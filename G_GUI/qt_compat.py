from __future__ import annotations

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    QT_API = "PyQt6"
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets

    QT_API = "PyQt5"

    class _EnumNamespace:
        def __new__(cls, value):
            return value

    class _ItemDataRole(_EnumNamespace):
        DisplayRole = QtCore.Qt.DisplayRole
        UserRole = QtCore.Qt.UserRole

    class _SortOrder(_EnumNamespace):
        AscendingOrder = QtCore.Qt.AscendingOrder
        DescendingOrder = QtCore.Qt.DescendingOrder

    class _FocusReason(_EnumNamespace):
        OtherFocusReason = QtCore.Qt.OtherFocusReason

    class _CursorShape(_EnumNamespace):
        PointingHandCursor = QtCore.Qt.PointingHandCursor

    class _DockWidgetArea(_EnumNamespace):
        RightDockWidgetArea = QtCore.Qt.RightDockWidgetArea

    class _Orientation(_EnumNamespace):
        Horizontal = QtCore.Qt.Horizontal
        Vertical = QtCore.Qt.Vertical

    class _AlignmentFlag(_EnumNamespace):
        AlignLeft = QtCore.Qt.AlignLeft
        AlignRight = QtCore.Qt.AlignRight
        AlignTrailing = QtCore.Qt.AlignTrailing
        AlignVCenter = QtCore.Qt.AlignVCenter

    class _ShortcutContext(_EnumNamespace):
        WindowShortcut = QtCore.Qt.WindowShortcut

    class _ColorRole(_EnumNamespace):
        Window = QtGui.QPalette.Window
        WindowText = QtGui.QPalette.WindowText
        Base = QtGui.QPalette.Base
        AlternateBase = QtGui.QPalette.AlternateBase
        ToolTipBase = QtGui.QPalette.ToolTipBase
        ToolTipText = QtGui.QPalette.ToolTipText
        Text = QtGui.QPalette.Text
        Button = QtGui.QPalette.Button
        ButtonText = QtGui.QPalette.ButtonText
        BrightText = QtGui.QPalette.BrightText
        Link = QtGui.QPalette.Link
        Highlight = QtGui.QPalette.Highlight
        HighlightedText = QtGui.QPalette.HighlightedText

    class _Policy(_EnumNamespace):
        Fixed = QtWidgets.QSizePolicy.Fixed
        Minimum = QtWidgets.QSizePolicy.Minimum
        Maximum = QtWidgets.QSizePolicy.Maximum
        Preferred = QtWidgets.QSizePolicy.Preferred
        Expanding = QtWidgets.QSizePolicy.Expanding
        MinimumExpanding = QtWidgets.QSizePolicy.MinimumExpanding
        Ignored = QtWidgets.QSizePolicy.Ignored

    class _IconMode(_EnumNamespace):
        Normal = QtGui.QIcon.Normal
        Disabled = QtGui.QIcon.Disabled
        Active = QtGui.QIcon.Active
        Selected = QtGui.QIcon.Selected

    class _IconState(_EnumNamespace):
        Off = QtGui.QIcon.Off
        On = QtGui.QIcon.On

    class _SelectionBehavior(_EnumNamespace):
        SelectRows = QtWidgets.QAbstractItemView.SelectRows

    QtCore.Qt.ItemDataRole = _ItemDataRole
    QtCore.Qt.SortOrder = _SortOrder
    QtCore.Qt.FocusReason = _FocusReason
    QtCore.Qt.CursorShape = _CursorShape
    QtCore.Qt.DockWidgetArea = _DockWidgetArea
    QtCore.Qt.Orientation = _Orientation
    QtCore.Qt.AlignmentFlag = _AlignmentFlag
    QtCore.Qt.ShortcutContext = _ShortcutContext
    QtGui.QPalette.ColorRole = _ColorRole
    QtGui.QAction = QtWidgets.QAction
    QtWidgets.QSizePolicy.Policy = _Policy
    QtGui.QIcon.Mode = _IconMode
    QtGui.QIcon.State = _IconState
    QtWidgets.QAbstractItemView.SelectionBehavior = _SelectionBehavior
