from PyQt5.QtWidgets import QWidget

from ..plugins.ptycho_fm import PtychoFMReconstructorLibrary
from .data import FileDialogFactory
from .parameters import ParameterViewBuilder
from .processing import ReconstructorViewControllerFactory


class PtychoFMViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self,
        model: PtychoFMReconstructorLibrary,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._model = model
        self._file_dialog_factory = file_dialog_factory

    @property
    def name(self) -> str:
        return 'PtychoFM'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        builder = ParameterViewBuilder(self._file_dialog_factory)
        enumerators = self._model.enumerators

        # Data
        data_group = 'Data'
        data_settings = self._model.data_settings
        builder.add_decimal_line_edit(
            data_settings.scale,
            'Diffraction Scale:',
            tool_tip='Scale factor applied to diffraction intensities before sqrt.',
            group=data_group,
        )
        builder.add_decimal_line_edit(
            data_settings.default_normalization,
            'Default Normalization:',
            group=data_group,
        )
        builder.add_check_box(
            data_settings.cache_object,
            'Cache Object In Memory',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.max_probe_modes,
            'Max Probe Modes:',
            tool_tip='Probes are zero-padded to this many mixed-state modes.',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.max_opr_modes,
            'Max OPR Modes:',
            tool_tip='Probes are zero-padded to this many orthogonal probe relaxation modes.',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.cache_memory_budget_mb,
            'Array Cache Budget (MiB):',
            tool_tip='Per-worker budget for cached object and probe arrays; 0 disables caching.',
            group=data_group,
        )
        builder.add_decimal_slider(
            data_settings.train_split,
            'Train Split:',
            tool_tip='Fraction of dataset used for training; remainder is validation.',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.random_seed,
            'Random Seed:',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.max_files,
            'Max Files (0 = no cap):',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.num_workers,
            'Dataloader Workers:',
            group=data_group,
        )
        builder.add_integer_line_edit(
            data_settings.prefetch_factor,
            'Prefetch Factor:',
            group=data_group,
        )

        # Model
        model_group = 'Model'
        model_settings = self._model.model_settings
        builder.add_combo_box(
            model_settings.encoder_type,
            enumerators.get_encoder_types(),
            'Encoder Type:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.img_size,
            'Image Size:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.patch_size,
            'Patch Size:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.embed_dim,
            'Embed Dimension:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.depth,
            'Depth:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.num_heads,
            'Attention Heads:',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.mlp_ratio,
            'MLP Ratio:',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.use_cls_token,
            'Use CLS Token',
            group=model_group,
        )
        builder.add_decimal_slider(
            model_settings.dropout,
            'Dropout:',
            group=model_group,
        )
        builder.add_decimal_slider(
            model_settings.attn_dropout,
            'Attention Dropout:',
            group=model_group,
        )
        builder.add_line_edit(
            model_settings.timm_model_name,
            'TIMM Model Name:',
            tool_tip="Only used when encoder_type is 'pretrained'.",
            group=model_group,
        )

        # Decoder
        builder.add_integer_line_edit(
            model_settings.decoder_base_channels,
            'Decoder Base Channels:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.decoder_latent_dim,
            'Decoder Latent Dim:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.decoder_num_stages,
            'Decoder Upsample Stages:',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.decoder_use_batchnorm,
            'Decoder Batch Norm',
            group=model_group,
        )
        builder.add_decimal_slider(
            model_settings.decoder_dropout,
            'Decoder Dropout:',
            group=model_group,
        )
        builder.add_combo_box(
            model_settings.coupled_decoder_mode,
            enumerators.get_coupled_decoder_modes(),
            'Coupled Decoder:',
            tool_tip=(
                "'none' decodes amplitude and phase separately; the coupled modes share one "
                'decoder and run the forward physics. A checkpoint loads only into the mode '
                'it was trained with.'
            ),
            group=model_group,
        )

        # Init
        builder.add_check_box(
            model_settings.init_enabled,
            'Custom Weight Init',
            group=model_group,
        )
        builder.add_combo_box(
            model_settings.init_method,
            enumerators.get_init_methods(),
            'Init Method:',
            group=model_group,
        )

        # Training
        training_group = 'Training'
        training_settings = self._model.training_settings
        builder.add_integer_line_edit(
            training_settings.epochs,
            'Epochs:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.batch_size,
            'Batch Size:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.learning_rate,
            'Learning Rate:',
            group=training_group,
        )
        builder.add_combo_box(
            training_settings.loss_function,
            enumerators.get_loss_functions(),
            'Loss Function:',
            group=training_group,
        )
        builder.add_combo_box(
            training_settings.weighted_loss_type,
            enumerators.get_loss_types(),
            'Weighted Loss Type:',
            tool_tip="Only used when loss_function is 'weighted'.",
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.weighted_loss_threshold,
            'Weighted Loss Threshold:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.weighted_loss_alpha,
            'Weighted Loss Alpha:',
            group=training_group,
        )
        builder.add_combo_box(
            training_settings.probe_aware_loss_type,
            enumerators.get_loss_types(),
            'Probe-Aware Loss Type:',
            tool_tip="Only used when loss_function is 'probe_aware'.",
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.probe_aware_loss_threshold,
            'Probe-Aware Loss Threshold:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.probe_aware_loss_alpha,
            'Probe-Aware Loss Alpha:',
            tool_tip='Power applied to the probe far-field envelope in the denominator.',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.probe_aware_loss_envelope_floor,
            'Probe-Aware Envelope Floor:',
            tool_tip='Per-sample floor as a fraction of the mean probe envelope.',
            group=training_group,
        )
        builder.add_combo_box(
            training_settings.q_dependent_loss_type,
            enumerators.get_loss_types(),
            'Q-Dependent Loss Type:',
            tool_tip="Only used when loss_function is 'q_dependent'.",
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.q_dependent_loss_alpha,
            'Q-Dependent Loss Alpha:',
            tool_tip='Focal exponent emphasizing poorly matched pixels; 0 disables it.',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.q_dependent_loss_q_beta,
            'Q-Dependent Q Exponent:',
            tool_tip='Radial exponent emphasizing high-Q pixels; 0 disables Q weighting.',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.q_dependent_loss_q_floor,
            'Q-Dependent Q Floor:',
            tool_tip='Minimum normalized Q, so low-Q pixels are not zero-weighted.',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.q_dependent_loss_threshold,
            'Q-Dependent Loss Threshold:',
            group=training_group,
        )
        builder.add_check_box(
            training_settings.q_dependent_loss_log_error,
            'Q-Dependent Log Error',
            tool_tip='Compute the data error on log10 amplitudes.',
            group=training_group,
        )
        builder.add_check_box(
            training_settings.q_dependent_loss_normalize_q,
            'Q-Dependent Normalize Q',
            tool_tip='Normalize the Q weight map to unit mean.',
            group=training_group,
        )
        builder.add_check_box(
            training_settings.save_epoch_models,
            'Save Per-Epoch Models',
            group=training_group,
        )

        # Inference
        inference_group = 'Inference'
        inference_settings = self._model.inference_settings
        builder.add_integer_line_edit(
            inference_settings.central_crop,
            'Central Crop:',
            tool_tip='Pixels cropped from each border of every patch before stitching.',
            group=inference_group,
        )
        builder.add_integer_line_edit(
            inference_settings.pad,
            'Fourier Shift Pad:',
            group=inference_group,
        )
        builder.add_integer_line_edit(
            inference_settings.batch_size,
            'Batch Size:',
            group=inference_group,
        )

        return builder.build_widget()
