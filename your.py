from TTS.tts.configs.shared_configs import BaseDatasetConfig
import os

output_path = "data"

dataset_config = BaseDatasetConfig(
    meta_file_train="metadata.csv", path=os.path.join(output_path, "bttm-out"), formatter="ljspeech"
)

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=250,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    save_step=500,
    lr=0.5
)


from TTS.utils.audio import AudioProcessor
ap = AudioProcessor.init_from_config(config)



from TTS.tts.utils.text.tokenizer import TTSTokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)


from TTS.tts.datasets import load_tts_samples
print(config.eval_split_size)
train_samples, eval_samples = load_tts_samples(
   	dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    # eval_split_size=1.0,
)

from TTS.tts.models.glow_tts import GlowTTS
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)


from trainer import Trainer, TrainerArgs
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)


trainer.fit()
