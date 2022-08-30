from dcase_models.data.features import Openl3 as Openl3_DM


class Openl3(Openl3_DM):
    def __init__(self, sequence_time=1.0, sequence_hop_time=0.5,
                 audio_win=1024, audio_hop=680, sr=22050,
                 content_type="env", input_repr="mel256", embedding_size=512):

        super().__init__(sequence_time=sequence_time,
                         sequence_hop_time=sequence_hop_time,
                         audio_win=audio_win, audio_hop=audio_hop,
                         content_type=content_type, input_repr=input_repr,
                         embedding_size=embedding_size)

    def calculate(self, file_name):
        import openl3
        audio = self.load_audio(file_name, change_sampling_rate=False)
        emb, ts = openl3.get_audio_embedding(
            audio, self.sr,
            model=self.openl3,
            hop_size=self.sequence_hop_time,
            center=False,
            verbose=False
        )

        return emb
