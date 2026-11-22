# Masks-to-Faces Generator

**Grupo 8** — Dani, David, Jiadong, Manuel y Brian.

Generación condicional de caras RGB a partir de máscaras de segmentación facial utilizando tres arquitecturas generativas entrenadas **desde cero**: **Difusión con U-Net**, **GAN condicional (Pix2Pix + pérdida perceptual)** y **Difusión con Transformer (DiT)**.

---

## Planteamiento y Dataset

El proyecto parte de un subconjunto de **CelebAMask-HQ** compuesto por **30.000 imágenes RGB de 64×64** y sus correspondientes **máscaras de segmentación binarias de 8 clases**:
* `hair` (pelo)
* `hat` (sombrero)
* `l_ear` / `r_ear` (orejas)
* `l_eye` / `r_eye` (ojos)
* `nose` (nariz)
* `mouth` (boca)

**Objetivo:** dada una máscara de segmentación de entrada, generar una cara realista y nueva que respete la geometría y posición de las facciones indicadas.

---

## Protocolo de Evaluación

Para comparar los modelos de forma justa se sigue un protocolo común:
* **Métricas:** **FID** (*Fréchet Inception Distance*) y **KID** (*Kernel Inception Distance*) calculados sobre las características de `Inception-v3`. En ambas métricas, **cuanto menor es el valor, mejor**.
* **Conjunto de test:** todos los modelos se evalúan generando **3.000 caras nuevas** a partir de las máscaras de test y comparándolas frente a las 3.000 caras reales de test.

---

## Arquitecturas Implementadas

* 1. Difusión condicional con U-Net (`diffusion_generator.ipynb`)
Modelo de difusión entrenado para predecir el ruido gaussiano $\epsilon$ añadido a la imagen en un paso temporal $t$.

* 2. GAN condicional con pérdida perceptual (`gan_generator.ipynb`)
Generación directa en un único paso hacia delante mediante entrenamiento adversarial condicionado a la máscara.

* 3. Difusión con DiT — *Diffusion Transformer* (`dit_generator.ipynb`)
Exploración que mantiene el mismo marco de difusión que la U-Net, pero reemplaza las convoluciones por un **Vision Transformer**.

---

## Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `diffusion_generator.ipynb` | Entrenamiento y evaluación del modelo de difusión condicional con U-Net. |
| `gan_generator.ipynb` | Entrenamiento y evaluación de la cGAN. |
| `dit_generator.ipynb` | Entrenamiento y evaluación del modelo de difusión basado en Transformer (DiT). |
| `models.txt` | Enlaces a los pesos preentrenados (para inferencia sin reentrenar) y al repositorio de GitHub. |
| `Enunciado.pdf` | Enunciado de la práctica. |