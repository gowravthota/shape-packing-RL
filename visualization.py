import sys
import math
from typing import List, Tuple, Optional, Union
from constraints import Shape, Container

# Optional pygame import with fallback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available, using matplotlib fallback for visualization")

# Matplotlib fallback for headless environments
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be limited")

# Configuration
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)
FPS = 60

def color_from_name(name: str) -> Union[tuple, str]:
    """Convert color name to RGB tuple or return as string for matplotlib."""
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
    }
    
    if PYGAME_AVAILABLE:
        try:
            return pygame.Color(name)
        except (ValueError, AttributeError):
            return color_map.get(name.lower(), (0, 0, 0))
    else:
        return name if name in color_map else "black"

def draw_shape_pygame(surface, shape: Shape, kind: str) -> None:
    """Draw shape using pygame."""
    if not PYGAME_AVAILABLE:
        raise ImportError("pygame is not available")
    
    color = color_from_name(shape.color)

    try:
        if kind == "rectangle":
            h = shape.size / 2
            pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
            coords = [shape._rotate(px, py) for px, py in pts]
            pygame.draw.polygon(surface, color, [(int(x), int(y)) for x, y in coords])

        elif kind == "circle":
            pygame.draw.circle(surface, color, (int(shape.x), int(shape.y)), int(shape.size / 2))

        elif kind == "triangle":
            r = shape.size / 2
            pts = [
                (0, -r),
                (r * math.cos(math.radians(210)), r * math.sin(math.radians(210))),
                (r * math.cos(math.radians(330)), r * math.sin(math.radians(330))),
            ]
            coords = [shape._rotate(px, py) for px, py in pts]
            pygame.draw.polygon(surface, color, [(int(x), int(y)) for x, y in coords])

        elif kind == "polygon":
            r = shape.size / 2
            pts = []
            for i in range(5):
                angle = math.radians(360 / 5 * i)
                px = r * math.cos(angle)
                py = r * math.sin(angle)
                pts.append((px, py))
            coords = [shape._rotate(px, py) for px, py in pts]
            pygame.draw.polygon(surface, color, [(int(x), int(y)) for x, y in coords])

        else:
            raise ValueError(f"Unknown shape kind: {kind}")
            
    except Exception as e:
        print(f"Error drawing shape: {e}")

def draw_shape_matplotlib(ax, shape: Shape, kind: str) -> None:
    """Draw shape using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is not available")
    
    color = color_from_name(shape.color)
    
    try:
        if kind == "rectangle":
            h = shape.size / 2
            rect = patches.Rectangle((shape.x - h, shape.y - h), shape.size, shape.size, 
                                   angle=shape.orientation, facecolor=color, alpha=0.7)
            ax.add_patch(rect)

        elif kind == "circle":
            circle = patches.Circle((shape.x, shape.y), shape.size / 2, 
                                  facecolor=color, alpha=0.7)
            ax.add_patch(circle)

        elif kind == "triangle":
            r = shape.size / 2
            pts = [
                (0, -r),
                (r * math.cos(math.radians(210)), r * math.sin(math.radians(210))),
                (r * math.cos(math.radians(330)), r * math.sin(math.radians(330))),
            ]
            coords = [shape._rotate(px, py) for px, py in pts]
            triangle = patches.Polygon(coords, facecolor=color, alpha=0.7)
            ax.add_patch(triangle)

        elif kind == "polygon":
            r = shape.size / 2
            pts = []
            for i in range(5):
                angle = math.radians(360 / 5 * i)
                px = r * math.cos(angle)
                py = r * math.sin(angle)
                pts.append((px, py))
            coords = [shape._rotate(px, py) for px, py in pts]
            polygon = patches.Polygon(coords, facecolor=color, alpha=0.7)
            ax.add_patch(polygon)

        else:
            raise ValueError(f"Unknown shape kind: {kind}")
            
    except Exception as e:
        print(f"Error drawing shape: {e}")

def visualize_container_matplotlib(container: Container, shapes: List[Tuple[Shape, str]], 
                                 title: str = "Container Visualization") -> None:
    """Visualize container and shapes using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Draw container
        container_rect = patches.Rectangle((container.x, container.y), 
                                         container.width, container.height,
                                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(container_rect)
        
        # Draw shapes
        for shape, kind in shapes:
            draw_shape_matplotlib(ax, shape, kind)
        
        # Set plot properties
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.show()
        
    except Exception as e:
        print(f"Error in matplotlib visualization: {e}")

def pygame_main():
    """Main pygame visualization loop."""
    if not PYGAME_AVAILABLE:
        print("pygame not available, using matplotlib fallback")
        return matplotlib_main()
    
    try:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Shape Container Visualization")
        clock = pygame.time.Clock()

        container = Container(450, 150, 300, 300, color="red")
        shapes = [
            (Shape(125, 125, orientation=0, size=80, color="blue"), "rectangle"),
            (Shape(125, 275, size=80, color="green"), "circle"),
            (Shape(125, 425, size=80, color="orange"), "triangle"),
            (Shape(125, 550, size=80, color="purple"), "polygon"),
        ]

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(BACKGROUND_COLOR)

            # Draw container
            pygame.draw.rect(
                screen,
                color_from_name(container.color),
                pygame.Rect(container.x, container.y, container.width, container.height),
                width=3,
            )

            # Draw shapes
            for shape, kind in shapes:
                draw_shape_pygame(screen, shape, kind)

            pygame.display.flip()
            clock.tick(FPS)

    except Exception as e:
        print(f"Error in pygame visualization: {e}")
    finally:
        if PYGAME_AVAILABLE:
            pygame.quit()

def matplotlib_main():
    """Main matplotlib visualization function."""
    container = Container(450, 150, 300, 300, color="red")
    shapes = [
        (Shape(125, 125, orientation=0, size=80, color="blue"), "rectangle"),
        (Shape(125, 275, size=80, color="green"), "circle"),
        (Shape(125, 425, size=80, color="orange"), "triangle"),
        (Shape(125, 550, size=80, color="purple"), "polygon"),
    ]
    
    visualize_container_matplotlib(container, shapes, "Shape Container Visualization")

def main():
    """Main function with fallback visualization options."""
    print("Space RL Visualization Module")
    print("Available backends:")
    print(f"  - pygame: {'✓' if PYGAME_AVAILABLE else '✗'}")
    print(f"  - matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'}")
    
    if PYGAME_AVAILABLE:
        pygame_main()
    elif MATPLOTLIB_AVAILABLE:
        matplotlib_main()
    else:
        print("No visualization backends available. Please install pygame or matplotlib.")

if __name__ == "__main__":
    main()