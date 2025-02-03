from __future__ import annotations
from dataclasses import dataclass
from typing import Final

from PyQt5.QtCore import QModelIndex, QPointF, QRectF, QSize, QSizeF, Qt
from PyQt5.QtGui import QColor, QFontMetrics, QPainter, QPen, QTextDocument, QTextOption
from PyQt5.QtWidgets import (
    QApplication,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
)

from ...model.agent import ChatMessageSender


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
        border_px: int = 3,
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
    BLUE_PEN: Final[QPen] = QPen(QColor('#243689'))
    BLUE_BRUSH: Final[QColor] = QColor('#0492d2')
    GREEN_PEN: Final[QPen] = QPen(QColor('#00894d'))
    GREEN_BRUSH: Final[QColor] = QColor('#78ca2a')

    def _create_text_document(
        self, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QTextDocument:
        text = index.model().data(index, Qt.DisplayRole)

        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WordWrap)
        text_option.setTextDirection(option.direction)

        doc = QTextDocument()
        doc.setDefaultTextOption(text_option)
        doc.setHtml(text)
        doc.setDefaultFont(option.font)
        doc.setDocumentMargin(0)
        doc.setTextWidth(self.TEXT_FRACTIONAL_WIDTH * option.rect.width())

        return doc

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        style = option.widget.style() if option.widget else QApplication.style()
        doc = self._create_text_document(option, index)
        metrics = BubbleMetrics.from_document(doc)
        sender = index.model().data(index, Qt.UserRole)

        # FIXME alignment/pen/brush to model
        alignment = Qt.AlignLeft
        bubble_pen = self.GREEN_PEN
        bubble_brush = self.GREEN_BRUSH

        if sender == ChatMessageSender.HUMAN:
            alignment = Qt.AlignRight
            bubble_pen = self.BLUE_PEN
            bubble_brush = self.BLUE_BRUSH

        bubble_pen.setWidth(metrics.border_px)

        doc_size = doc.size()
        item_size = QSizeF(
            doc_size.width() + 2 * metrics.mbp_px,
            doc_size.height() + 2 * metrics.mbp_px,
        )
        layout_rect = QStyle.alignedRect(
            Qt.LayoutDirectionAuto,
            alignment,
            item_size.toSize(),
            style.subElementRect(QStyle.SE_ItemViewItemText, option),
        )
        bubble_origin = QPointF(
            layout_rect.left() + metrics.margin_px,
            layout_rect.top() + metrics.margin_px,
        )
        bubble_rect = QRectF(
            0.0,
            0.0,
            doc_size.width() + 2 * metrics.bp_px,
            doc_size.height() + 2 * metrics.bp_px,
        )
        text_rect = QRectF(
            0.0,
            0.0,
            doc_size.width(),
            doc_size.height(),
        )

        # Painting item without text (this takes care of painting e.g. the highlighted for selected
        # or hovered over items in an ItemView)
        style.drawControl(QStyle.CE_ItemViewItem, option, painter, option.widget)

        painter.save()
        painter.setPen(bubble_pen)
        painter.setBrush(bubble_brush)
        painter.translate(bubble_origin)
        painter.drawRoundedRect(bubble_rect, metrics.radius_px, metrics.radius_px)
        painter.translate(metrics.bp_px, metrics.bp_px)
        doc.drawContents(painter, text_rect)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        doc = self._create_text_document(option, index)
        metrics = BubbleMetrics.from_document(doc)
        hint = option.rect.size()
        hint.setHeight(int(doc.size().height()) + 2 * metrics.mbp_px)
        return hint
