# Third-party attribution

The unchanged pretrained VideoMAE weights originate from MCG-NJU/videomae-base-finetuned-kinetics, revision 488eb9a0565f257b32866000305c8178965eb9f6:
https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics

The model card identifies its license as CC BY-NC 4.0 (attribution, noncommercial): https://creativecommons.org/licenses/by-nc/4.0/ . This project is an academic COMP6116 demonstration. Redistribution here does not relicense the pretrained weights or grant commercial-use rights. Credit: Zhan Tong et al., “VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training,” 2022, https://github.com/MCG-NJU/VideoMAE .

The pretrained encoder is unchanged; our rectangular/variable-length input adapter and AnimalAct heads are described in the technical reports. Runtime pooling/sampling rules and corresponding modifications are documented in the root README.

Raw AnimalAct course videos, annotation files, and demo data are NOT distributed in this public repository. Use course-provided data under its original terms. Third-party Python dependencies retain their respective licenses. Publishing source here does not establish a blanket license for all embedded research materials.
