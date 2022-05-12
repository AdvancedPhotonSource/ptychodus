
        zlf = lambda x: numpy.zeros_like(x, dtype=float)

        view.imageRibbon.scalarTransformComboBox.addItem('Identity',
                                                         ScalarTransformation(lambda x: x))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Square Root',
            ScalarTransformation(lambda x: numpy.sqrt(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Logarithm (Base 2)',
            ScalarTransformation(lambda x: numpy.log2(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Natural Logarithm',
            ScalarTransformation(lambda x: numpy.log(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Logarithm (Base 10)',
            ScalarTransformation(lambda x: numpy.log10(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.currentTextChanged.connect(
            lambda text: controller._renderCachedImageData())

        view.imageRibbon.complexComponentComboBox.addItem(
            'Magnitude', ComplexToRealStrategy(numpy.absolute, False))
        view.imageRibbon.complexComponentComboBox.addItem('Phase',
                                                          ComplexToRealStrategy(numpy.angle, True))
        view.imageRibbon.complexComponentComboBox.addItem('Real Part',
                                                          ComplexToRealStrategy(numpy.real, False))
        view.imageRibbon.complexComponentComboBox.addItem('Imaginary Part',
                                                          ComplexToRealStrategy(numpy.imag, False))
        view.imageRibbon.complexComponentComboBox.currentTextChanged.connect(
            lambda text: controller._renderCachedImageData())
