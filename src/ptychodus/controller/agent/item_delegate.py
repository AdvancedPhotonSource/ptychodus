from __future__ import annotations
from dataclasses import dataclass
from typing import Final

from PyQt5.QtCore import QModelIndex, QPointF, QRectF, QSize, QSizeF, Qt
from PyQt5.QtGui import QBrush, QFontMetrics, QPainter, QPen, QTextDocument, QTextOption
from PyQt5.QtWidgets import QApplication, QStyle, QStyleOptionViewItem, QStyledItemDelegate


@dataclass(frozen=True)
class BubbleMetrics:
    margin_px: int
    border_px: int
    padding_px: int
    radius_px: int

    @property
    def mbp_px(self) -> int:
        return self.margin_px + self.border_px + self.padding_px

    @property
    def bp_px(self) -> int:
        return self.margin_px + self.border_px

    @classmethod
    def from_document(
        cls,
        document: QTextDocument,
        *,
        margin_em: int = 1,
        border_px: int = 1,
        padding_em: int = 1,
        radius_px: int = 10,
    ) -> BubbleMetrics:
        font_metrics = QFontMetrics(document.defaultFont())
        one_em_px = font_metrics.horizontalAdvance('m')
        return cls(
            margin_px=margin_em * one_em_px,
            border_px=border_px,
            padding_px=padding_em * one_em_px,
            radius_px=radius_px,
        )


class ChatBubbleItemDelegate(QStyledItemDelegate):
    TEXT_FRACTIONAL_WIDTH: Final[float] = 0.8

    def _create_text_document(
        self, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QTextDocument:
        text = index.data(Qt.ItemDataRole.DisplayRole)

        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WrapMode.WordWrap)
        text_option.setTextDirection(option.direction)

        doc = QTextDocument()
        doc.setDefaultTextOption(text_option)
        doc.setHtml(text)
        doc.setDefaultFont(option.font)
        doc.setDocumentMargin(0)

        text_width = min(self.TEXT_FRACTIONAL_WIDTH * option.rect.width(), doc.idealWidth())
        doc.setTextWidth(text_width)

        return doc

    def paint(
        self, painter: QPainter | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if painter is None:
            return

        style = option.widget.style() if option.widget else QApplication.style()

        if style is None:
            raise ValueError('style is None!')

        doc = self._create_text_document(option, index)
        metrics = BubbleMetrics.from_document(doc)
        alignment = Qt.Alignment(index.data(Qt.ItemDataRole.TextAlignmentRole))

        doc_size = doc.size()
        item_size = QSizeF(
            doc_size.width() + 2 * metrics.mbp_px,
            doc_size.height() + 2 * metrics.mbp_px,
        )
        layout_rect = QStyle.alignedRect(
            Qt.LayoutDirection.LayoutDirectionAuto,
            alignment,
            item_size.toSize(),
            style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, option),
        )
        bubble_rect = QRectF(
            layout_rect.left() + metrics.margin_px,
            layout_rect.top() + metrics.margin_px,
            doc_size.width() + 2 * metrics.bp_px,
            doc_size.height() + 2 * metrics.bp_px,
        )
        text_origin = QPointF(
            bubble_rect.left() + metrics.bp_px,
            bubble_rect.top() + metrics.bp_px,
        )
        text_rect = QRectF(
            0.0,
            0.0,
            doc_size.width(),
            doc_size.height(),
        )

        bubble_brush = QBrush(index.data(Qt.ItemDataRole.BackgroundRole))
        bubble_pen = QPen(index.data(Qt.ItemDataRole.ForegroundRole))
        bubble_pen.setWidth(metrics.border_px)

        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)

        painter.save()
        painter.setPen(bubble_pen)
        painter.setBrush(bubble_brush)
        painter.drawRoundedRect(bubble_rect, metrics.radius_px, metrics.radius_px)
        painter.translate(text_origin)
        doc.drawContents(painter, text_rect)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        doc = self._create_text_document(option, index)
        metrics = BubbleMetrics.from_document(doc)
        hint = option.rect.size()
        hint.setHeight(int(doc.size().height()) + 2 * metrics.mbp_px)
        return hint
