from PyQt5.QtWidgets import QWidget

from ..model.ptycho_fm.core import PtychoFMReconstructorLibrary
from .data import FileDialogFactory
from .parametric import ParameterViewBuilder
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
            data_settings.packed,
            'Packed Dataset',
            tool_tip='Use pre-packed dataset shards instead of paired HDF5 files.',
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
            data_settings.target_size,
            'Target Pattern Size:',
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
        builder.add_combo_box(
            data_settings.sharding_strategy,
            enumerators.get_sharding_strategies(),
            'Sharding Strategy:',
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
        builder.add_check_box(
            data_settings.use_cuda_prefetcher,
            'Use CUDA Prefetcher',
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
        builder.add_combo_box(
            training_settings.mode,
            enumerators.get_training_modes(),
            'Mode:',
            group=training_group,
        )
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
            enumerators.get_weighted_loss_types(),
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
        builder.add_integer_line_edit(
            training_settings.validation_plot_freq,
            'Validation Plot Freq:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.checkpoint_freq,
            'Checkpoint Freq:',
            group=training_group,
        )
        builder.add_check_box(
            training_settings.save_epoch_models,
            'Save Per-Epoch Models',
            group=training_group,
        )
        builder.add_check_box(
            training_settings.resume_from_checkpoint,
            'Resume From Checkpoint',
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
