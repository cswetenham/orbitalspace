
#include "orbitalSpaceApp.h"

#include <SFML/Window.hpp>

int __cdecl main()
{
    sf::ContextSettings settings;
    settings.depthBits         = 24; // Request a 24 bits depth buffer
    settings.stencilBits       = 8;  // Request a 8 bits stencil buffer
    settings.antialiasingLevel = 2;  // Request 2 levels of antialiasing
    sf::Window window(sf::VideoMode(800, 600, 32), "SFML OpenGL", sf::Style::Close, settings);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Some code for stopping application on close or when escape is pressed...
          if (event.type == sf::Event::Resized)
          {
            glViewport(0, 0, event.size.width, event.size.height);
          }

          if (event.type == sf::Event::Closed)
          {
            window.close();
          }
        }

        window.display();
    }

    return 0;
}
