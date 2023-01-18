# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, List, Union

from modelscope.metainfo import Preprocessors
from modelscope.models.base import Model
from modelscope.utils.constant import Fields, Frameworks
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = ['WavToScp']


@PREPROCESSORS.register_module(
    Fields.audio, module_name=Preprocessors.wav_to_scp)
class WavToScp(Preprocessor):
    """generate audio scp from wave or ark
    """

    def __init__(self):
        pass

    def __call__(self,
                 model: Model = None,
                 recog_type: str = None,
                 audio_format: str = None,
                 audio_in: Union[str, bytes] = None,
                 audio_fs: int = None) -> Dict[str, Any]:
        assert model is not None, 'preprocess model is empty'
        assert recog_type is not None and len(
            recog_type) > 0, 'preprocess recog_type is empty'
        assert audio_format is not None, 'preprocess audio_format is empty'
        assert audio_in is not None, 'preprocess audio_in is empty'

        self.am_model = model
        out = self.forward(self.am_model.forward(), recog_type, audio_format,
                           audio_in, audio_fs)
        return out

    def forward(self, model: Dict[str, Any], recog_type: str,
                audio_format: str, audio_in: Union[str, bytes], audio_fs: int,
                cmd: Dict[str, Any]) -> Dict[str, Any]:
        assert len(recog_type) > 0, 'preprocess recog_type is empty'
        assert len(audio_format) > 0, 'preprocess audio_format is empty'
        assert len(
            model['am_model']) > 0, 'preprocess model[am_model] is empty'
        assert len(model['am_model_path']
                   ) > 0, 'preprocess model[am_model_path] is empty'
        assert os.path.exists(
            model['am_model_path']), 'preprocess am_model_path does not exist'
        assert len(model['model_workspace']
                   ) > 0, 'preprocess model[model_workspace] is empty'
        assert os.path.exists(model['model_workspace']
                              ), 'preprocess model_workspace does not exist'
        assert len(model['model_config']
                   ) > 0, 'preprocess model[model_config] is empty'

        cmd['model_workspace'] = model['model_workspace']
        cmd['am_model'] = model['am_model']
        cmd['am_model_path'] = model['am_model_path']
        cmd['recog_type'] = recog_type
        cmd['audio_format'] = audio_format
        cmd['model_config'] = model['model_config']
        cmd['audio_fs'] = audio_fs
        if 'code_base' in cmd['model_config']:
            code_base = cmd['model_config']['code_base']
        else:
            code_base = None

        if isinstance(audio_in, str):
            # wav file path or the dataset path
            cmd['wav_path'] = audio_in
        if code_base != 'funasr':
            cmd = self.config_checking(cmd)
        cmd = self.env_setting(cmd)
        if audio_format == 'wav':
            cmd['audio_lists'] = self.scp_generation_from_wav(cmd)
        elif audio_format == 'kaldi_ark':
            cmd['audio_lists'] = self.scp_generation_from_ark(cmd)
        elif audio_format == 'tfrecord':
            cmd['audio_lists'] = os.path.join(cmd['wav_path'], 'data.records')
        elif audio_format == 'pcm' or audio_format == 'scp':
            cmd['audio_lists'] = audio_in

        return cmd

    def config_checking(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """config checking
        """

        assert inputs['model_config'].__contains__(
            'type'), 'model type does not exist'
        inputs['model_type'] = inputs['model_config']['type']
        # code base
        if 'code_base' in inputs['model_config']:
            code_base = inputs['model_config']['code_base']
        else:
            code_base = None
        inputs['code_base'] = code_base
        # decoding mode
        if 'mode' in inputs['model_config']:
            mode = inputs['model_config']['mode']
        else:
            mode = None
        inputs['mode'] = mode

        if inputs['model_type'] == Frameworks.torch:
            assert inputs['model_config'].__contains__(
                'batch_size'), 'batch_size does not exist'

            if inputs['model_config'].__contains__('am_model_config'):
                am_model_config = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['am_model_config'])
                assert os.path.exists(
                    am_model_config), 'am_model_config does not exist'
                inputs['am_model_config'] = am_model_config
            else:
                inputs['am_model_config'] = ''
            if inputs['model_config'].__contains__('asr_model_config'):
                asr_model_config = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['asr_model_config'])
                assert os.path.exists(
                    asr_model_config), 'asr_model_config does not exist'
                inputs['asr_model_config'] = asr_model_config
            else:
                asr_model_config = ''
                inputs['asr_model_config'] = ''

            if 'asr_model_wav_config' in inputs['model_config']:
                asr_model_wav_config: str = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['asr_model_wav_config'])
                assert os.path.exists(asr_model_wav_config
                                      ), 'asr_model_wav_config does not exist'
            else:
                asr_model_wav_config: str = inputs['asr_model_config']

            # the lm model file path
            if 'lm_model_name' in inputs['model_config']:
                lm_model_path = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['lm_model_name'])
            else:
                lm_model_path = None
            # the lm config file path
            if 'lm_model_config' in inputs['model_config']:
                lm_model_config = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['lm_model_config'])
            else:
                lm_model_config = None
            if lm_model_path and lm_model_config and os.path.exists(
                    lm_model_path) and os.path.exists(lm_model_config):
                inputs['lm_model_path'] = lm_model_path
                inputs['lm_model_config'] = lm_model_config
            else:
                inputs['lm_model_path'] = None
                inputs['lm_model_config'] = None
            if 'audio_format' in inputs:
                if inputs['audio_format'] == 'wav' or inputs[
                        'audio_format'] == 'pcm':
                    inputs['asr_model_config'] = asr_model_wav_config
                else:
                    inputs['asr_model_config'] = asr_model_config

            if inputs['model_config'].__contains__('mvn_file'):
                mvn_file = os.path.join(inputs['model_workspace'],
                                        inputs['model_config']['mvn_file'])
                assert os.path.exists(mvn_file), 'mvn_file does not exist'
                inputs['mvn_file'] = mvn_file
            if inputs['model_config'].__contains__('vad_model_name'):
                vad_model_name = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['vad_model_name'])
                assert os.path.exists(
                    vad_model_name), 'vad_model_name does not exist'
                inputs['vad_model_name'] = vad_model_name
            if inputs['model_config'].__contains__('vad_model_config'):
                vad_model_config = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['vad_model_config'])
                assert os.path.exists(
                    vad_model_config), 'vad_model_config does not exist'
                inputs['vad_model_config'] = vad_model_config
            if inputs['model_config'].__contains__('vad_mvn_file'):
                vad_mvn_file = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['vad_mvn_file'])
                assert os.path.exists(
                    vad_mvn_file), 'vad_mvn_file does not exist'
                inputs['vad_mvn_file'] = vad_mvn_file
            if inputs['model_config'].__contains__('punc_model_name'):
                punc_model_name = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['punc_model_name'])
                assert os.path.exists(
                    punc_model_name), 'punc_model_name does not exist'
                inputs['punc_model_name'] = punc_model_name
            if inputs['model_config'].__contains__('punc_model_config'):
                punc_model_config = os.path.join(
                    inputs['model_workspace'],
                    inputs['model_config']['punc_model_config'])
                assert os.path.exists(
                    punc_model_config), 'punc_model_config does not exist'
                inputs['punc_model_config'] = punc_model_config

        elif inputs['model_type'] == Frameworks.tf:
            assert inputs['model_config'].__contains__(
                'vocab_file'), 'vocab_file does not exist'
            vocab_file: str = os.path.join(
                inputs['model_workspace'],
                inputs['model_config']['vocab_file'])
            assert os.path.exists(vocab_file), 'vocab file does not exist'
            inputs['vocab_file'] = vocab_file

            assert inputs['model_config'].__contains__(
                'am_mvn_file'), 'am_mvn_file does not exist'
            am_mvn_file: str = os.path.join(
                inputs['model_workspace'],
                inputs['model_config']['am_mvn_file'])
            assert os.path.exists(am_mvn_file), 'am mvn file does not exist'
            inputs['am_mvn_file'] = am_mvn_file

        else:
            raise ValueError('model type is mismatched')

        return inputs

    def env_setting(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # run with datasets, should set datasets_path and text_path
        if inputs['recog_type'] != 'wav':
            inputs['datasets_path'] = inputs['wav_path']

            # run with datasets, and audio format is waveform
            if inputs['audio_format'] == 'wav':
                inputs['wav_path'] = os.path.join(inputs['datasets_path'],
                                                  'wav', inputs['recog_type'])
                inputs['reference_text'] = os.path.join(
                    inputs['datasets_path'], 'transcript', 'data.text')
                assert os.path.exists(
                    inputs['reference_text']), 'reference text does not exist'

            # run with datasets, and audio format is kaldi_ark
            elif inputs['audio_format'] == 'kaldi_ark':
                inputs['wav_path'] = os.path.join(inputs['datasets_path'],
                                                  inputs['recog_type'])
                inputs['reference_text'] = os.path.join(
                    inputs['wav_path'], 'data.text')
                assert os.path.exists(
                    inputs['reference_text']), 'reference text does not exist'

            # run with datasets, and audio format is tfrecord
            elif inputs['audio_format'] == 'tfrecord':
                inputs['wav_path'] = os.path.join(inputs['datasets_path'],
                                                  inputs['recog_type'])
                inputs['reference_text'] = os.path.join(
                    inputs['wav_path'], 'data.txt')
                assert os.path.exists(
                    inputs['reference_text']), 'reference text does not exist'
                inputs['idx_text'] = os.path.join(inputs['wav_path'],
                                                  'data.idx')
                assert os.path.exists(
                    inputs['idx_text']), 'idx text does not exist'

        # set asr model language
        if 'lang' in inputs['model_config']:
            inputs['model_lang'] = inputs['model_config']['lang']
        else:
            inputs['model_lang'] = 'zh-cn'

        return inputs

    def scp_generation_from_wav(self, inputs: Dict[str, Any]) -> List[Any]:
        """scp generation from waveform files
        """
        from easyasr.common import asr_utils

        # find all waveform files
        wav_list = []
        if inputs['recog_type'] == 'wav':
            file_path = inputs['wav_path']
            if os.path.isfile(file_path):
                if file_path.endswith('.wav') or file_path.endswith('.WAV'):
                    wav_list.append(file_path)
        else:
            wav_dir: str = inputs['wav_path']
            wav_list = asr_utils.recursion_dir_all_wav(wav_list, wav_dir)

        list_count: int = len(wav_list)
        inputs['wav_count'] = list_count

        # store all wav into audio list
        audio_lists = []
        j: int = 0
        while j < list_count:
            wav_file = wav_list[j]
            wave_key: str = os.path.splitext(os.path.basename(wav_file))[0]
            item = {'key': wave_key, 'file': wav_file}
            audio_lists.append(item)
            j += 1

        return audio_lists

    def scp_generation_from_ark(self, inputs: Dict[str, Any]) -> List[Any]:
        """scp generation from kaldi ark file
        """

        ark_scp_path = os.path.join(inputs['wav_path'], 'data.scp')
        ark_file_path = os.path.join(inputs['wav_path'], 'data.ark')
        assert os.path.exists(ark_scp_path), 'data.scp does not exist'
        assert os.path.exists(ark_file_path), 'data.ark does not exist'

        with open(ark_scp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # store all ark item into audio list
        audio_lists = []
        for line in lines:
            outs = line.strip().split(' ')
            if len(outs) == 2:
                key = outs[0]
                sub = outs[1].split(':')
                if len(sub) == 2:
                    nums = sub[1]
                    content = ark_file_path + ':' + nums
                    item = {'key': key, 'file': content}
                    audio_lists.append(item)

        return audio_lists
