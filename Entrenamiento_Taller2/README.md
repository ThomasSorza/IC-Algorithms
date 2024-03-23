# IC-Algorithms - Taller 2 Entrenamiento de un modelo de ML 

## Ejecución de Algoritmos de Machine Learning

Este repositorio contiene implementaciones de algoritmos de Machine Learning en Python. Asegúrate de tener instalados Python y pip para ejecutar los algoritmos.

## Descarga y uso del proyecto:
| Descripción                                     | Enlace de Descarga                                            |
|-------------------------------------------------|---------------------------------------------------------------|
| Usa GitDown para descargar el repositorio completo: | [Git Down](https://minhaskamal.github.io/DownGit/#/home) |
| Repositorio de Algoritmos en GitHub             | [GitHub - Algoritmos](https://github.com/sh4dex/IC-Algorithms/tree/main/Entrenamiento) |

>Posteriormente se descarga un archivo .zip desde el navegador el cual se debe descomprimir para poder ejecutar los scripts .py

## Instalación de Dependencias

Para instalar las dependencias necesarias, puedes usar el archivo `requirements.txt`. Ejecuta el siguiente comando en tu terminal para instalar todas las dependencias:

```bash
pip install -r requirements.txt
```

## Descripción de las columnas en el excel de ataques de phising:

| **Característica**                   | **Definición**                                                                                   |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| NumDots                         | Número de puntos en la URL.                                                                    |
| SubdomainLevel                  | Nivel del subdominio en la URL.                                                                |
| PathLevel                       | Nivel del camino en la URL.                                                                    |
| UrlLength                       | Longitud de la URL.                                                                            |
| NumDash                         | Número de guiones en la URL.                                                                   |
| NumDashInHostname               | Número de guiones en el nombre del host de la URL.                                              |
| AtSymbol                        | Indicador de presencia del símbolo "@" en la URL.                                               |
| TildeSymbol                     | Indicador de presencia del símbolo "~" en la URL.                                               |
| NumUnderscore                   | Número de guiones bajos en la URL.                                                             |
| NumPercent                      | Número de símbolos de porcentaje (%) en la URL.                                                 |
| NumQueryComponents              | Número de componentes de consulta en la URL.                                                    |
| NumAmpersand                    | Número de símbolos de ampersand (&) en la URL.                                                  |
| NumHash                         | Número de símbolos de almohadilla (#) en la URL.                                                |
| NumNumericChars                 | Número de caracteres numéricos en la URL.                                                       |
| NoHttps                         | Indicador de ausencia de HTTPS en la URL.                                                       |
| RandomString                    | Indicador de presencia de una cadena aleatoria en la URL.                                       |
| IpAddress                       | Indicador de presencia de una dirección IP en la URL.                                           |
| DomainInSubdomains              | Indicador de presencia de un dominio en los subdominios de la URL.                              |
| DomainInPaths                   | Indicador de presencia de un dominio en las rutas de la URL.                                     |
| HttpsInHostname                 | Indicador de presencia de HTTPS en el nombre del host de la URL.                                |
| HostnameLength                  | Longitud del nombre de host de la URL.                                                          |
| PathLength                      | Longitud de la ruta de la URL.                                                                 |
| QueryLength                     | Longitud de la consulta de la URL.                                                              |
| DoubleSlashInPath               | Indicador de presencia de doble barra en la ruta de la URL.                                      |
| NumSensitiveWords               | Número de palabras sensibles en la URL.                                                         |
| EmbeddedBrandName               | Indicador de presencia de un nombre de marca incrustado en la URL.                              |
| PctExtHyperlinks                | Porcentaje de hipervínculos externos en la URL.                                                 |
| PctExtResourceUrls              | Porcentaje de URL de recursos externos en la URL.                                               |
| ExtFavicon                      | Indicador de presencia de un favicon externo en la URL.                                          |
| InsecureForms                   | Indicador de presencia de formularios no seguros en la URL.                                      |
| RelativeFormAction              | Indicador de presencia de acciones de formulario relativas en la URL.                            |
| ExtFormAction                   | Indicador de presencia de acciones de formulario externas en la URL.                              |
| AbnormalFormAction              | Indicador de presencia de acciones de formulario anormales en la URL.                            |
| PctNullSelfRedirectHyperlinks   | Porcentaje de hipervínculos de auto-redirección nulos en la URL.                                 |
| FrequentDomainNameMismatch      | Indicador de frecuentes discrepancias en el nombre de dominio en la URL.                         |
| FakeLinkInStatusBar             | Indicador de enlaces falsos en la barra de estado de la URL.                                     |
| RightClickDisabled              | Indicador de deshabilitación del clic derecho en la URL.                                         |
| PopUpWindow                     | Indicador de presencia de ventanas emergentes en la URL.                                         |
| SubmitInfoToEmail               | Indicador de envío de información a través de correo electrónico en la URL.                      |
| IframeOrFrame                   | Indicador de presencia de iframes o frames en la URL.                                            |
| MissingTitle                    | Indicador de título faltante en la URL.                                                         |
| ImagesOnlyInForm                | Indicador de imágenes únicamente en formularios en la URL.                                       |
| SubdomainLevelRT                | Característica temporal relacionada con el nivel del subdominio.                                  |
| UrlLengthRT                     | Característica temporal relacionada con la longitud de la URL.                                    |
| PctExtResourceUrlsRT            | Característica temporal relacionada con el porcentaje de URL de recursos externos.                |
| AbnormalExtFormActionR          | Característica temporal relacionada con acciones de formulario externas anormales.                |
| ExtMetaScriptLinkRT             | Característica temporal relacionada con meta scripts y enlaces externos.                          |
| PctExtNullSelfRedirectHyperlinksRT | Característica temporal relacionada con el porcentaje de hipervínculos de auto-redirección nulos.|
| CLASS_LABEL                     | Etiqueta de clase que indica si la URL es phishing o legítima.                                   |
