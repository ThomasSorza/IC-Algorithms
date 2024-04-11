# IC-Algorithms - Taller 4 Contar palabras

## Ejecución del script para contar palabras

## Descarga y uso del proyecto:
| Descripción                                     | Enlace de Descarga                                            |
|-------------------------------------------------|---------------------------------------------------------------|
| Usa GitDown para descargar el repositorio completo: | [Git Down](https://minhaskamal.github.io/DownGit/#/home) |
| Repositorio de Algoritmos en GitHub             | [GitHub - Algoritmos](https://github.com/sh4dex/IC-Algorithms/tree/main/Word_Count_Algorithm_Taller4) |

>Posteriormente se descarga un archivo .zip desde el navegador el cual se debe descomprimir para poder ejecutar los word_count.py

## Validación bash

Tenemos que tener instaldo bash, con el siguiente comando podemos tener la ruta del archivo binario de bash o zsh.
```bash
wich bash
```
para zsh
```bash
wich zsh
```
Output experado:
```bash
/usr/bin/zsh
/usr/bin/bash
```
Recomendable trabajar en ambientes GNU/Linux, MacOs o basados en Unix.
## Perms

Como se trata de un archivo ejecutable se requiere los permisos necesarios para ejecutar este script.

```bash
chmod +x word_count.sh
```

Realizado el levantamiento de permisos debemos ejecutar el archivo, flag ***-n*** para *normalizar el texto*, flag ***-w*** contar palabras seguido del archivo con el texto que queremos utilizar.

```bash
sh word_count.sh -[n|w] file.txt
#ó
./word_cound.sh -[n|w] file.txt
```