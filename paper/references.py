"""AMA reference entries for the manuscript, keyed by the \\cite keys in main.tex.

Titles, authors, volumes and pages were checked on 12 Oct 2026 against PubMed (entries with a PMID), the arXiv
abstract page, Crossref (IEEE entries) or the publisher page. Entries added on 21 Oct 2026
(ahmed2026, adnan2022, ziller2021, tayebi2024, pati2024 and the three datasets) were checked against PubMed,
Crossref or DataCite the same day. build.py orders the list by first citation.
LaTeX markup is allowed. Journal abbreviations follow the NLM catalogue.
"""
REFS = {
    "sheller2019": "Sheller MJ, Reina GA, Edwards B, Martin J, Bakas S. Multi-institutional deep learning modeling "
                   "without sharing patient data: a feasibility study on brain tumor segmentation. "
                   "\\textit{Brainlesion}. 2019;11383:92-104. doi:10.1007/978-3-030-11723-8\\_9",
    "sheller2020": "Sheller MJ, Edwards B, Reina GA, et al. Federated learning in medicine: facilitating "
                   "multi-institutional collaborations without sharing patient data. \\textit{Sci Rep}. "
                   "2020;10(1):12598. doi:10.1038/s41598-020-69250-1",
    "pati2022": "Pati S, Baid U, Edwards B, et al. Federated learning enables big data for rare cancer boundary "
                "detection. \\textit{Nat Commun}. 2022;13(1):7346. doi:10.1038/s41467-022-33407-5",
    "albalawi2024": "Albalawi E, T R M, Thakur A, et al. Integrated approach of federated learning with transfer "
                    "learning for classification and diagnosis of brain tumor. \\textit{BMC Med Imaging}. "
                    "2024;24(1):110. doi:10.1186/s12880-024-01261-0",
    "mastoi2025": "Mastoi QU, Latif S, Brohi S, et al. Explainable AI in medical imaging: an interpretable and "
                  "collaborative federated learning model for brain tumor classification. \\textit{Front Oncol}. "
                  "2025;15:1535478. doi:10.3389/fonc.2025.1535478",
    "muntaqim2025": "Muntaqim MDZ, Smrity TA. Federated learning framework for brain tumor detection using MRI images "
                    "in non-IID data distributions. \\textit{J Imaging Inform Med}. 2025;38(6):3909-3927. "
                    "doi:10.1007/s10278-025-01484-9",
    "paul2026": "Paul M, Tiwari A, Barmashe B, Rai K, Panwar VS. Federated MobileNetV2 with ensemble meta-learning "
                "for privacy-preserving brain tumor classification. \\textit{Sci Rep}. 2026;16(1). "
                "doi:10.1038/s41598-026-55847-5",
    "vangala2026": "Vangala BP, Guda S, Peddi L, Vangala N. Scores that hold, benchmarks that leak: measuring dataset "
                   "contamination in public brain-tumor MRI classification. \\textit{arXiv}. Preprint posted online "
                   "September 30, 2026. arXiv:2610.00421",
    "lakshmi2026": "Lakshmi Vasanthi K, Sree Darshne J, Venkatasubbu P, Ramasubramanian P. A federated multimodal deep "
                   "learning framework for brain tumor classification using MRI. \\textit{Front Artif Intell}. "
                   "2026;9:1754000. doi:10.3389/frai.2026.1754000",
    "qazi2026": "Qazi EUH, Al-Ghanem WK, Faheem MH, Ullah H. Federated learning framework for privacy-preserving "
                "explainable AI-driven clinical decision-making. \\textit{IEEE J Biomed Health Inform}. Published "
                "online 2026. doi:10.1109/JBHI.2026.3679499",
    "abadi2016": "Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy. In: \\textit{Proceedings "
                 "of the 2016 ACM SIGSAC Conference on Computer and Communications Security}. ACM; 2016:308-318. "
                 "doi:10.1145/2976749.2978318",
    "gupta2026": "Gupta S, Gupta M, Kumar R, Abraham A. A federated learning and explainable AI framework for "
                 "privacy-preserving brain tumor diagnosis using multi-institutional MRI data. \\textit{IEEE Access}. "
                 "2026;14:22630-22643. doi:10.1109/ACCESS.2026.3660305",
    "kaissis2021": "Kaissis G, Ziller A, Passerat-Palmbach J, et al. End-to-end privacy preserving deep learning on "
                   "multi-institutional medical imaging. \\textit{Nat Mach Intell}. 2021;3(6):473-484. "
                   "doi:10.1038/s42256-021-00337-8",
    "ziller2024": "Ziller A, Mueller TT, Stieger S, et al. Reconciling privacy and accuracy in AI for medical imaging. "
                  "\\textit{Nat Mach Intell}. 2024;6(7):764-774. doi:10.1038/s42256-024-00858-y",
    "knolle2026": "Knolle MA, Menten MJ, Jungmann F, et al. Disparate privacy risks from medical AI. \\textit{Nature}. "
                  "2026;656(8126):192-198. doi:10.1038/s41586-026-10688-0",
    "saifullah2024": "Saifullah S, Mercier D, Lucieri A, Dengel A, Ahmed S. The privacy-explainability trade-off: "
                     "unraveling the impacts of differential privacy and federated learning on attribution methods. "
                     "\\textit{Front Artif Intell}. 2024;7:1236947. doi:10.3389/frai.2024.1236947",
    "manna2024": "Manna S, Sett N. Reconciling privacy and explainability in high-stakes: a systematic inquiry. "
                 "\\textit{arXiv}. Preprint posted online December 30, 2024. arXiv:2412.20798",
    "cheng2015": "Cheng J, Huang W, Cao S, et al. Enhanced performance of brain tumor classification via tumor region "
                 "augmentation and partition. \\textit{PLoS One}. 2015;10(10):e0140381. "
                 "doi:10.1371/journal.pone.0140381",
    "hsu2019": "Hsu TMH, Qi H, Brown M. Measuring the effects of non-identical data distribution for federated visual "
               "classification. \\textit{arXiv}. Preprint posted online 2019. arXiv:1909.06335",
    "he2016": "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: \\textit{Proceedings of "
              "the IEEE Conference on Computer Vision and Pattern Recognition}. IEEE; 2016:770-778. "
              "doi:10.1109/CVPR.2016.90",
    "mcmahan2017": "McMahan HB, Moore E, Ramage D, Hampson S, Ag\\\"uera y Arcas B. Communication-efficient learning of "
                   "deep networks from decentralized data. In: \\textit{Proceedings of the 20th International "
                   "Conference on Artificial Intelligence and Statistics}. PMLR; 2017;54:1273-1282.",
    "mironov2017": "Mironov I. R\\'enyi differential privacy. In: \\textit{Proceedings of the 30th IEEE Computer "
                   "Security Foundations Symposium}. IEEE; 2017:263-275. doi:10.1109/CSF.2017.11",
    "yousefpour2021": "Yousefpour A, Shilov I, Sablayrolles A, et al. Opacus: user-friendly differential privacy "
                      "library in PyTorch. \\textit{arXiv}. Preprint posted online 2021. "
                      "arXiv:2109.12298",
    "papernot2022": "Papernot N, Steinke T. Hyperparameter tuning with R\\'enyi differential privacy. In: "
                    "\\textit{International Conference on Learning Representations}. 2022. arXiv:2110.03620",
    "dwork2014": "Dwork C, Roth A. The algorithmic foundations of differential privacy. \\textit{Found Trends Theor "
                 "Comput Sci}. 2014;9:211-407. doi:10.1561/0400000042",
    "mcmahan2018": "McMahan HB, Ramage D, Talwar K, Zhang L. Learning differentially private recurrent language "
                   "models. \\textit{arXiv}. Preprint posted online 2017. arXiv:1710.06963",
    "li2020": "Li T, Sahu AK, Zaheer M, Sanjabi M, Talwalkar A, Smith V. Federated optimization in heterogeneous "
              "networks. In: \\textit{Proceedings of Machine Learning and Systems}. 2020;2:429-450.",
    "karimireddy2020": "Karimireddy SP, Kale S, Mohri M, Reddi S, Stich S, Suresh AT. SCAFFOLD: stochastic "
                       "controlled averaging for federated learning. In: \\textit{Proceedings of the 37th "
                       "International Conference on Machine Learning}. PMLR; 2020;119:5132-5143.",
    "carlini2022": "Carlini N, Chien S, Nasr M, Song S, Terzis A, Tram\\`er F. Membership inference attacks from first "
                   "principles. In: \\textit{2022 IEEE Symposium on Security and Privacy}. IEEE; 2022:1897-1914. "
                   "doi:10.1109/SP46214.2022.9833649",
    "selvaraju2017": "Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations "
                     "from deep networks via gradient-based localization. In: \\textit{Proceedings of the IEEE "
                     "International Conference on Computer Vision}. IEEE; 2017:618-626. doi:10.1109/ICCV.2017.74",
    "chattopadhay2018": "Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. Grad-CAM++: generalized "
                        "gradient-based visual explanations for deep convolutional networks. In: \\textit{2018 IEEE "
                        "Winter Conference on Applications of Computer Vision}. IEEE; 2018:839-847. "
                        "doi:10.1109/WACV.2018.00097",
    "sundararajan2017": "Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. In: "
                        "\\textit{Proceedings of the 34th International Conference on Machine Learning}. PMLR; "
                        "2017;70:3319-3328.",
    "petsiuk2018": "Petsiuk V, Das A, Saenko K. RISE: randomized input sampling for explanation of black-box models. "
                   "In: \\textit{Proceedings of the British Machine Vision Conference}. 2018. arXiv:1806.07421",
    "adebayo2018": "Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. In: "
                   "\\textit{Advances in Neural Information Processing Systems 31}. 2018. arXiv:1810.03292",
    "ahmed2026": "Ahmed MM, Adebayo UO, Othman ZK, et al. Artificial intelligence in neuro-oncology imaging: "
                 "advancing brain tumor detection, grading, and treatment response evaluation. \\textit{Intell Oncol}. "
                 "2026;2(3):100071. doi:10.1016/j.intonc.2026.100071",
    "jung2026": "Jung JH, Sunwoo L, Ji H, Yoo S, Park JE, Lee JG. Deep learning-based segmentation of small-volume brain "
                "metastases in lung cancer patients. \\textit{Intell Oncol}. 2026;2(2):100049. "
                "doi:10.1016/j.intonc.2026.100049",
    "li2026": "Li D, Wei J, L\\\"{u} Q, Le J, Liao X. Privacy-preserving medical image processing based on cryptography "
              "primitives: a review of methods. \\textit{Intell Oncol}. 2026;2(4):100083. doi:10.1016/j.intonc.2026.100083",
    "cai2025": "Cai W. DeepSeek AI: transforming medical AI with cost-efficiency, transparency, and privacy preservation. "
               "\\textit{Intell Oncol}. 2025;1(2):172-175. doi:10.1016/j.intonc.2025.03.006",
    "adnan2022": "Adnan M, Kalra S, Cresswell JC, Taylor GW, Tizhoosh HR. Federated learning and differential privacy "
                 "for medical image analysis. \\textit{Sci Rep}. 2022;12(1):1953. doi:10.1038/s41598-022-05539-7",
    "ziller2021": "Ziller A, Usynin D, Braren R, Makowski M, Rueckert D, Kaissis G. Medical imaging deep learning with "
                  "differential privacy. \\textit{Sci Rep}. 2021;11(1):13524. doi:10.1038/s41598-021-93030-0",
    "tayebi2024": "Tayebi Arasteh S, Ziller A, Kuhl C, et al. Preserving fairness and diagnostic accuracy in private "
                  "large-scale AI models for medical imaging. \\textit{Commun Med (Lond)}. 2024;4(1):46. "
                  "doi:10.1038/s43856-024-00462-6",
    "pati2024": "Pati S, Kumar S, Varma A, et al. Privacy preservation for federated learning in health care. "
                "\\textit{Patterns (N Y)}. 2024;5(7):100974. doi:10.1016/j.patter.2024.100974",
    "chengdata": "[dataset] Cheng J. Brain tumor dataset. figshare. Version 8. 2024. "
                 "doi:10.6084/m9.figshare.1512427.v8",
    "sartajdata": "[dataset] Bhuvaji S, Kadam A, Bhumkar P, Dedge S, Kanchan S. Brain tumor classification (MRI). "
                  "Kaggle. Version 3. 2020. doi:10.34740/kaggle/dsv/1183165",
    "br35hdata": "[dataset] Hamada A. Br35H :: Brain tumor detection 2020. Kaggle. Version 12. Accessed October 1, 2026. "
                 "\\url{https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection}",
}
