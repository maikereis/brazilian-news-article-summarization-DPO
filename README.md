# Brazilian News Article Summarization DPO

This repository contains the dataset and code for the **Brazilian News Article Summarization DPO**, a dataset created to fine-tune language models using **Direct Preference Optimization (DPO)** for the task of summarizing Brazilian news articles.

The dataset is publicly available on [**Hugging Face**](https://huggingface.co/datasets/maikerdr/brazilian-news-article-summarization-DPO)

## Dataset Overview

This dataset contains human preference data for pairs of Brazilian Portuguese news summaries. It adopts a DPO format, where each sample includes:

- `article`: The original Brazilian news article.
- `triplets`: A list of preference comparisons, each with:
  - `instruction`: The instruction or prompt.
  - `chosen`: The preferred summary.
  - `rejected`: The less preferred summary.

This format supports training models using DPO, PPO/RLHF, or reward modeling approaches.

---

## Example Sample

Each entry in the dataset looks like this:

```json
{
  "id": "b5fd4541‑094d‑4096‑83d6‑548b9ec01837",
  "article": "Bandeira dos EUA Foto: Reuters/Vincent Alban … Como tirar o visto de turismo para os EUA Veja como tirar o passaporte",
  "triplets": [
    {
      "id": "c4b954a6‑7e63‑4da8‑a84a‑49914ea509ce",
      "instruction": "Resumo da novidade sobre prazos de renovação do visto.",
      "chosen": "Os Estados Unidos reduziram temporariamente o prazo para solicitar a renovação do visto de não‑imigrante para 12 meses, sem a necessidade de realização de entrevistas em consulados.",
      "rejected": "As alterações foram anunciadas pelos profissionais de assessoria de viagens e têm sido divulgadas nas últimas semanas."
    },
    {
      "id": "3590c402‑ccf9‑4824‑ad6a‑4dd92c6c0656",
      "instruction": "Quais são as condições para renovação do visto de não‑imigrante?",
      "chosen": "Para se qualificar para uma renovação de visto (com ou sem entrevista), é necessário possuir certas características específicas, como se poder ser dispensado da entrevista e estar atualmente com seu visto expirado.",
      "rejected": "As novidades mencionadas envolvem alterações ao prazo de renovação do visto, mas especificamente detalharam as condições para tal procedimento."
    },
    {
      "id": "cdd33de4‑d4e4‑408b‑8cb1‑9be642200ea5",
      "instruction": "Qual é a diferença entre os tipos de visto J1 e J2?",
      "chosen": "Os visados J1 e J2 (para estudantes, professores, pesquisadores e seus dependentes) podem sofrer alterações, com a validade sendo ampliada de 12 para 24 meses e uma taxa adicional aplicável a eles será cobrada.",
      "rejected": "O texto informa que os tipos de visto J1 e J2 têm validades diferentes, mas não especifica suas diferenças concretas."
    }
  ]
}
```

## Install

  pip install -r requirements.txt

## Load Dataset
```python
  from datasets import load_dataset
  
  ds = load_dataset("maikerdr/brazilian-news-article-summarization‑DPO")
  print(ds["train"][0])
```

## Recommended Usage

Ideal for:

- Preference‑based fine‑tuning (DPO, PPO/RLHF)

- Reward modeling

- Supervised summarization in Portuguese

Direct Preference Optimization (DPO) trains models by focusing directly on human preferences without the need for complex reward models or reinforcement learning

