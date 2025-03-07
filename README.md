# Checklist

- [*] Accuracy of Idefics2  
- [ ] Llava Implementation  
- [ ] Other Datasets  
- [ ] Other Models  
- [ ] GPT API for Candidate Datasets and Candidate Models  



# README: Comparison of ICL and Fine-Tuned Idefics2 on VisMin Benchmark

## Overview
This document provides a comparison between **ICL (In-Context Learning) results** and the **fine-tuned version of Idefics2** on the VisMin benchmark. The analysis is based on **Text (T), Image (I), and Group (G) scores** for four key categories: **Object, Attribute, Spatial Relation, and Count**.

---

## Performance Summary
### **1. Text Scores (T)**
- ICL performs **very well in text-based reasoning**, often matching or slightly trailing behind the fine-tuned version.
- Fine-tuned Idefics2 still has a slight edge, but **ICL remains competitive** in object and attribute understanding.

### **2. Image Scores (I)**
- **Fine-tuned Idefics2 massively outperforms ICL in image-based understanding.**
- The ICL version struggles with recognizing fine-grained visual changes.
- The biggest improvement in fine-tuning is seen in **Object (+52 points), Attribute (+57 points), and Count (+15 points)**.

### **3. Group Scores (G)**
- **Fine-tuning significantly enhances the model's holistic multimodal reasoning.**
- ICL **performs poorly** in Spatial Relation and Count categories, while the fine-tuned model achieves much higher accuracy.
- The **Group score gap is largest in the Count category**, where the fine-tuned model has **over a 40-point lead**.

---

## **Detailed Comparison Table**
| Category     | Text Score (T) - ICL | Text Score (T) - Fine-Tuned | Image Score (I) - ICL | Image Score (I) - Fine-Tuned | Group Score (G) - ICL | Group Score (G) - Fine-Tuned |
|-------------|----------------------|----------------------------|----------------------|----------------------------|----------------------|----------------------------|
| **Object**      | 94.65                | 95.4                        | 17.44               | 69.4                        | 17.10               | 67.6                        |
| **Attribute**   | 89.12                | 89.1                        | 14.29               | 71.4                        | 12.59               | 67.0                        |
| **S. Relation** | 42.28                | 18.6                        | 13.99               | 18.8                        | 6.59                | 4.8                         |
| **Count**       | 77.25                | 72.2                        | 35.14               | 50.6                        | 30.73               | 47.0                        |
| **Overall Avg.**| **37.60**             | **55.99**                    | **-**                | **-**                        | **-**                | **-**                        |

---

## **Key Takeaways**
1. **ICL is strong in text-based reasoning** but struggles with multimodal understanding.
2. **Fine-tuning significantly improves visual reasoning**, especially in Object, Attribute, and Count categories.
3. **Spatial Relation remains a challenge for both models**, with **ICL performing slightly better in text understanding**, but still struggling in image-based reasoning.
4. **Fine-tuning provides a substantial boost to Image and Group scores**, making it far more capable of understanding fine-grained multimodal relationships.
5. **ICL's improved performance in Count reasoning** (higher than fine-tuned in Text score) suggests some capability in numerical reasoning, but still lags behind when integrating image understanding.

---

## **Conclusion**
- **ICL is good for text-based tasks but underperforms in visual grounding.**
- **Fine-tuning is essential for achieving strong multimodal performance.**
- **Future improvements should focus on enhancing spatial reasoning and count-based understanding** to close the remaining performance gaps.

This README serves as a summary of the key findings and performance trends between **ICL and Fine-tuned Idefics2** on the **VisMin Benchmark**.


