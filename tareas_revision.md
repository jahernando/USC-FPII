# Revisión Jupyter Book — Neutrinos USC
Fecha: Marzo 2026

---

## Estructura general
- [ ] El Book abre correctamente en el navegador (_build/html/index.html)
- [ ] Navegación entre los 6 capítulos funciona (links Next/Previous)
- [ ] Links cruzados entre notebooks funcionan
- [ ] nu_index — mapa conceptual y tabla de capítulos son correctos

---

## nu_sm.ipynb
- [ ] Flujo histórico correcto: Pauli → Cowan-Reines → V-A → LEP
- [ ] Figuras y referencias correctas

---

## nu_oscillations.ipynb (teoría + experimentos clásicos)
- [ ] El notebook termina con el forward pointer a nu_oscillations_exp
- [ ] Widget interactivo de 2 familias funciona (celda 19)
- [ ] Widget interactivo PMNS de 3 familias funciona (celda 104)
- [ ] Sección MSW — fórmulas y diagrama correctos
- [ ] Resultados SNO, SK solar y atmosférico — números actualizados

---

## nu_oscillations_exp.ipynb (precisión + nueva generación)
- [ ] Resultados Daya Bay, RENO, DoubleChooz — límites de theta_13 correctos
- [ ] Sección NOvA: análisis 10 años (septiembre 2025) — números correctos
- [ ] Sección T2K+NOvA joint (octubre 2025) — referencia arXiv:2510.19888
- [ ] Widget JUNO funciona (oscillations.exercise_juno_smearing())
- [ ] Primeros resultados JUNO (noviembre 2025) — 59 días, arXiv:2511.14593
- [ ] Sección HyperKamiokande — excavación completada julio 2025
- [ ] Sección ORCA/ARCA/IceCube — números y referencias correctos
    - [ ] ORCA primer resultado (2024): sin²θ23=0.51, arXiv:2406.08588
    - [ ] IceCube NGC 1068: arXiv:2209.04519
    - [ ] IceCube plano galáctico: arXiv:2307.04427
- [ ] Summary actualizado con resultados 2025

---

## nu_mass.ipynb (nuevo)
- [ ] Tabla de splittings NuFit-6.0 — valores correctos
- [ ] Sección KATRIN — límite 0.45 eV, referencia arXiv:2406.13516
- [ ] Ejercicio KATRIN funciona (majorana.exercise_katrin_endpoint())
- [ ] Sección DESI 2024 — Σmν < 0.072 eV, referencia arXiv:2404.03002
- [ ] Tabla resumen de constraints — coherencia entre filas
- [ ] Sección Project 8 — suficientemente precisa o ampliar

---

## nu_majorana.ipynb
- [ ] Sección seesaw (Tipo I/II/III) — fórmulas y texto correctos
- [ ] Sección leptogénesis — mecanismo Fukugita-Yanagida, conexión con delta_CP
- [ ] Sección nEXO corregida — ya no dice "will not see the light"
- [ ] Los 5 ejercicios interactivos del módulo majorana.py funcionan
- [ ] Perspectivas LEGEND-1000, KamLAND2-Zen, NEXT-HD — números y timelines
- [ ] KATRIN fue eliminado correctamente (ahora solo en nu_mass)
- [ ] Conclusiones reflejan el estado actual 2025

---

## Referencias
- [ ] intro.md — PDG 2024 link funciona
- [ ] nu_mass.ipynb — referencias KATRIN [K1][K2], Planck [P18], DESI [D24], NuFit [NF6]
- [ ] nu_majorana.ipynb — referencia LEGEND-1000 CDR [L1K] arXiv:2107.11462
- [ ] Referencias [38]-[44] en nu_oscillations_exp (JUNO, T2K+NOvA) correctas

---

## Idioma y estilo
- [ ] Títulos de sección consistentes en inglés en todos los notebooks
- [ ] Secciones JUNO y DUNE con texto en español — decidir si unificar
- [ ] Summary de nu_oscillations_exp (en español) correcto y completo

---

## Prioridad alta (contenido nuevo)
- [ ] Seesaw Tipo I/II/III en nu_majorana — revisar con detalle
- [ ] Leptogénesis en nu_majorana — revisar con detalle
- [ ] nu_mass.ipynb completo — es nuevo, revisar todo
- [ ] Sección ORCA/ARCA/IceCube — es nueva, revisar todo
