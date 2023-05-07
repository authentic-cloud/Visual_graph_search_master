# Visual Graph Search

Este es un juego que permite visualizar conceptos y jugar con algoritmos de búsqueda para estructuras de datos con grafos.
La interfaz está implementada con PyGame, permite dibujar muros y seleccionar las posiciones iniciales y finales.

![](images/demo.gif)

Referencia de colores:

- ![#000000](https://via.placeholder.com/15/000000/000000?text=+) Celda vacía
- ![#404040](https://via.placeholder.com/15/404040/000000?text=+) Celda de muro
- ![#FF0000](https://via.placeholder.com/15/ff0000/000000?text=+) Celda origen
- ![#00FF00](https://via.placeholder.com/15/00ff00/000000?text=+) Celda objetivo
- ![#808040](https://via.placeholder.com/15/808040/000000?text=+) Celda activa (el algoritmo de búsqueda está analizando este nodo)
- ![#808080](https://via.placeholder.com/15/808080/000000?text=+) Celda explorada (el algoritmo de búsqueda analizó todos los nodos hijos de esta celda)
- ![#FFFF00](https://via.placeholder.com/15/ffff00/000000?text=+) Celda camino (el mejor camino estimado por el algoritmo)

Actualmente soporta:

| Nombre                   |              | Tipo              | Wikipedia link                                      |
|--------------------------|--------------|-------------------|-----------------------------------------------------|
| Depth-first search       | `DFS`        | Uninformed search | https://en.wikipedia.org/wiki/Depth-first_search    |
| Breath-first search      | `BFS`        | Uninformed search | https://en.wikipedia.org/wiki/Breadth-first_search  |
| Greedy best-first search | `GREEDY_BFS` | Informed search   | https://en.wikipedia.org/wiki/Best-first_search     |
| A* search                | `A_STAR`     | Informed search   | https://en.wikipedia.org/wiki/A*_search_algorithm   |

## Comenzando

Estas instrucciones te van a dejar una copia del projecto funcionando en tu máquina para propósitos de desarrollo y pruebas.

### Prerrequisitos

Asegurate de tener instalado Python 3.7 o superior.

### Instalación

Clona este repositorio e instala los requerimientos:
```
$ git clone https://github.com/nahueespinosa/visual_search.git
$ pip install -r requirements.txt
```

### Uso

Ejecuta el comando:
```
$ python runner.py
```

## Ideas

Hay muchas pruebas que se pueden hacer. Aquí hay un ejemplo usando otro tamaño de tablero.

![](images/no_wall.gif)
![](images/wall.gif)

Sentite libre de descargar y experimentar con el código.

## Detalles

Los datos son representados internamente como un grafo para resolver por el algoritmo.
Cada nodo es un estado en el sistema y el objetivo es encontrar un camino entre el estado inicial y el final.

![](images/graph.png)

En el archivo `solver.py` puedes encontrar una implementación abstracta de los algoritmos para usar en otro proyectos.
Existe sólo una clase `Solver` y diferentes tipos de fronteras generan diferentes comportamientos.

```python
 # The algorithm selected determines the type of frontier to be used
frontier = StackFrontier() if algorithm == self.DFS else \
           QueueFrontier() if algorithm == self.BFS else \
           GreedyFrontier(lambda node: self.environment.cost_to_target(node.state)) if algorithm == self.GREEDY_BFS else \
           GreedyFrontier(lambda node: self.environment.cost_to_target(node.state) + node.cost_from_source)
```

La clase `StackFrontier` siempre expande el nodo más profundo. El nodo más profundo es el último que fue agregado.

La clase `QueueFrontier` siempre expande el nodo superficial. El nodo superficial
es el primero que fue agregado la lista.

La clase `GreedyFrontier` siempre expande el nodo que está más cerca del objetivo, estimado por una función heurística (`cost_function`).
Lo hace agregandolos en orden usando la función de coste.

## Reconocimientos

Este proyecto fue inspirado por el curso ["Introduction to Artificial Intelligence with Python" of CS50](https://cs50.harvard.edu/ai/2020/).
