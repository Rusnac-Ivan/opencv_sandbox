#ifndef _EVENT_HANDLER_
#define _EVENT_HANDLER_

#include "Platform.h"

class View;

class EventHandler
{
private:
	static View *mView;

	EventHandler() = delete;
	EventHandler(EventHandler &) = delete;
	EventHandler &operator=(EventHandler &) = delete;
	~EventHandler() = delete;

public:
	static void SetListener(View* view);

	static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
	static void MouseMoveCallback(GLFWwindow *window, double xpos, double ypos);
	static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
	static void MouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset);
	static void FramebufferSizeCallback(GLFWwindow *window, int width, int height);

private:
};

#endif