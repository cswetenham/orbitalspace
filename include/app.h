/* 
 * File:   app.h
 * Author: fib
 *
 * Created on 19 December 2011, 12:06
 */

#ifndef APP_H
#define	APP_H

#include "timer.h"

#ifdef _WIN32
# include <Windows.h>
#endif

#include "orStd.h"
#include "orMath.h"
#include "orGfx.h"

#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

class App
{
public:
  App();
  virtual ~App();

protected:
  virtual void InitRender();
  virtual void ShutdownRender();

  virtual void InitState();
  virtual void ShutdownState();
  
  virtual void HandleEvent(sf::Event const& _event) = 0;
  
  virtual void UpdateState(double const _dt) = 0;
  
  virtual void RenderState() = 0;

public:  
  virtual void Run();
  
  void Init();
  void Shutdown();

protected:
  void PollEvents();
  void BeginRender();
  void EndRender();
    
protected:
  sf::RenderWindow* m_window;
  Timer::PerfTime m_lastFrameDuration;
  bool m_running;
};

#endif	/* APP_H */

