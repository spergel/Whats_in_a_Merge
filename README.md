# Whats_in_a_Merge


#Interseting Papers:

Mixture of Experts

[Prompt-prompted Mixture of Experts for Efficient LLM Generation](https://arxiv.org/abs/2404.01365)

[LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs](https://arxiv.org/abs/2401.16160)

[OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2402.01739)

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

[Mixture of Experts Explained](https://huggingface.co/blog/moe)

[Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)

[StableMoE: Stable Routing Strategy for Mixture of Experts](https://arxiv.org/abs/2204.08396)

#Translation:

[An In-depth Look at Gemini’s Language Abilities](https://arxiv.org/abs/2312.11444)

[Large Language Models Are State-of-the-Art Evaluators of Translation Quality](https://arxiv.org/abs/2302.14520)

[How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation](https://arxiv.org/abs/2302.09210)

General Objectives:

1. Establish a GitHub repository to organize the current codebase and related research. Seek out additional resources specific to router training.
2. Review MergeKit's algorithm for generating expert routers and assess its applicability for our analysis needs.
3. Reach out to Arcee AI, the creators of MergeKit, to explore potential collaboration opportunities. (or Lucas Atkins, a prominent extender of Mergekit)
4. Write code that shows router confusion.
5. Design a visually engaging, color-coded matrix to illustrate how different prompts are processed by a specific router.
6. Generate substantial amounts of router confusion data for different cases (e.g. input text taken directly from fine tuning data used for each expert, text we expect to be very difficult for the router to classify, non-english text, etc)
7. ideally finding an interesting result
8. write paper outlining results

Translation-Specific Goals:

1. Initiate trials with Claude Haiku for translation purposes, bearing in mind the potential costs.
2. Establish a comprehensive translations database.
3. Proceed with the fine-tuning of the SeaLLM 7b models.
4. Integrate the fine-tuned models.
5. Address the general objectives outlined above.


