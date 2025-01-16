# Applying object-oriented principles in Godot

The engine offers two main ways to create reusable objects: scripts and scenes.

==Neither of these technically define classes under the hood.==

Still, many best practices using Godot involve applying object-oriented programming principles to the scripts and scenes that compose your game.

That is why it's useful to understand how we can think of them as classes.

This guide briefly explains how scripts and scenes work in the engine's core to help you understand how they work under the hood.


## How scripts work in the engine

The engine provides built-in classes like Node. You can extend those to create derived types using a script.

These scripts are not technically classes. Instead, they are resources that tell the engine a sequence of initializations to perform on one of the engine's built-in classes.

Godot's internal classes have methods that register a class's data with a ClassDB. This database provides runtime access to class information.

ClassDB contains information about classes like:
- Properties
- Methods
- Constants
- Signals




## Scenes

