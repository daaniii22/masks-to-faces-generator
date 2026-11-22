# masks-to-faces-generator

Grupo 8 · Dani, David, Jiadong, Manuel y Brian.

Hemos generado caras a partir de máscaras de segmentación con modelos entrenados desde cero.


Hemos partido del dataset CelebAMask-HQ, en el subconjunto de la asignatura: 30.000 caras a color de 64×64 y sus máscaras de segmentación de 8 clases (pelo, sombrero, orejas, ojos, nariz, boca).
La entrada del modelo es una máscara y la salida una cara nueva que respete esa máscara.
La arquitectura tiene que poder condicionarse con la máscara (GAN, modelos de difusión o transformers), entrenarse desde cero y sin fine-tuning.

¿Como evaluamos?
FID y KID con las características de Inception-v3, entre las caras generadas y caras reales. Cuanto menos, mejor.
Siempre con máscaras de test, que el modelo no vio al entrenar. Los tres modelos se miden con 3.000 caras de test a color.
Suelo: el FID entre dos grupos de caras reales (7,6). Es lo mejor que podría sacar un modelo perfecto.
Los FID que damos son los de "caras nuevas": generadas a partir de las máscaras de test y comparadas con las caras reales de test.

Los tres modelos que hemos realizado son:

Difusión condicional con U-Net (DDPM + DDIM): FID 15,5 y KID 0,0064.
GAN condicional con pérdida perceptual VGG: FID 49,4 y KID 0,043.
Difusión con DiT (Diffusion Transformer): FID 31,2 y KID 0,020.
1. Difusión condicional con U-Net 
Entrada: la imagen con ruido x_t concatenada con la máscara (64×64×11).
U-Net con filtros 64·128·256·256, bloques residuales con GroupNorm + SiLU y atención de 4 cabezas en las resoluciones 16×16 y 8×8.
Ruido: T = 1000 pasos con cosine schedule; el paso t pasa por un embedding sinusoidal y un MLP hasta 256 dimensiones, y se inyecta en cada bloque residual con FiLM (escala y desplazamiento).
Pérdida: MSE sobre el ruido ε. 9,4 M de parámetros, 100 épocas.
Generación con muestreo DDIM (η = 1). En validación, 100 pasos dieron FID 37,4 y 10 pasos 50,5.
2. GAN condicional con pérdida perceptual 
Generador: U-Net con filtros 64·128·256·512·512, InstanceNorm y upsampling bilineal + convolución 3×3.
Discriminador: PatchGAN con Spectral Norm que da un mapa de 16×16.
Pérdida del generador: 3·LSGAN + 20·L1 + 0,08·VGG16 (pérdida perceptual).
13,2 M de parámetros, 30 épocas.
3. Difusión con DiT 
Misma difusión que la U-Net  (T = 1000, cosine schedule, pérdida MSE sobre ε, DDIM); solo cambia la red que predice el ruido, que pasa de convoluciones a un Transformer.

Sin encoder ni decoder ni skips: la entrada de 11 canales (imagen con ruido + 8 máscaras one-hot, que no reciben ruido) se divide en parches de 4×4 (patchify), lo que da 256 tokens de dimensión 256. Los tokens mantienen ese tamaño en los 8 bloques y al final se vuelven a juntar en una imagen de 64×64×3.
Cada bloque usa atención global (4 cabezas) y un MLP. El paso t entra por adaLN-Zero: genera la escala, el desplazamiento y un gate que se inicializa a cero, así cada bloque empieza siendo la identidad. La posición de los parches se codifica con un embedding sin-cos 2D fijo.
9,79 M de parámetros. AdamW (lr 2e-4, weight decay 1e-4) con cosine decay, batch de 64, hasta 100 épocas, y copia EMA de los pesos (0,999).
Se usan los pesos EMA de la mejor época en validación (la 70, FID de validación 50,6).

Para generar se compararon varias configuraciones de DDIM en validación. La mejor fue 10 pasos con η = 0 (FID de validación 50,06), y es la que se usa en test.
Con el mismo protocolo que los dos modelos , el DiT queda por detrás de la difusión con U-Net (31,2 frente a 15,5). Se planteó como una prueba de una arquitectura distinta, no como el modelo principal. Los dos no se diferencian solo en la red: el DiT usa EMA y otro muestreo (η = 0), y la U-Net no.

Qué hay en el repositorio

-diffusion_generator.ipynb: difusión con U-Net.

-gan_generator.ipynb: GAN con pérdida perceptual.

-dit_generator.ipynb: DIT

-models.txt: enlace a los modelos ya entrenados, para no reentrenar.

-Enunciado.pdf: enunciado de la práctica.
