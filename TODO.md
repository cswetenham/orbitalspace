# TODO

- IMGui camera sub-widget with position, look pos / orientation, FOV, far/near
- IMGui timeline with scrolling, zooming
  - Look at imgui_demo Horizontal Scrolling, Graph Widgets examples
  - Might use ImGui::PlotEx directly
  - Want to draw a resizing scale at the bottom, maybe some full-height lines at
    larger period
  - Want to zoom in/out (+/- buttons for now?)
  - Persistent (always visible) (scrollable?) column on left side with list of objects. Different sort orders? (earliest maneuver, distance, name, faction?)
  - Persistent (always visible) scale with ticks and times (mission time? absolute? relative? toggle button?)
  - Timeline per active object
  - Indicators for maneuvers (include non-0 length maneuvers??)
  - Hover maneuver for info
  - click timeline or maneuver to show 'ghost' future positions
  - Add maneuver buttons in menu will pick the appropriate time
  - Some auto-maneuvers will include sliders e.g. bi-elliptic transfer height
  - right-click timeline to add custom maneuver at future time for controlled objects
  - Show delta-v for units
  - alternative timeline view showing total remaining delta-v over time instead of maneuvers chart

  - DONE basic layout, actual timelines missing
  - TODO draw timeline
  - TODO interactivity
  - TODO displaying patched conics? Or we could display future trajectory
    relative to current SOI, or patch at SOI changes and maneuvers?