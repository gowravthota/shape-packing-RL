import sys
import math
import pygame
from constraints import Shape, Container

pygame.init()

WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)
FPS = 60

def color_from_name(name: str) -> pygame.Color:
    try:
        return pygame.Color(name)
    except ValueError:
        return pygame.Color("black")


def draw_shape(surface: pygame.Surface, shape: Shape, kind: str) -> None:
    color = color_from_name(shape.color)

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


def main():
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

        pygame.draw.rect(
            screen,
            color_from_name(container.color),
            pygame.Rect(container.x, container.y, container.width, container.height),
            width=3,
        )

        for shape, kind in shapes:
            draw_shape(screen, shape, kind)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()