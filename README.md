# Import

Du importierst das Projekt bei Pycharm mit VSC/GIT -> clone (url: https://github.com/OliverStoll/MLL-1/ , ordner: wo du es drin haben willst (solltest neuen erstellen))

Dafür musst du dich in Pycharm eventuell in Git einloggen, schaffste.

# Aufbau

Der Code ist in `/stubs`, die Tests sind in `/tests`.

Wenn du den Code testen willst kannst du die Tests einfach ganz normal ausführen und bekommst ein Interface was dir sagt welche Tests fehlschlagen.

# Debuggen

Wenn du wissen möchtest warum, bzw. welche falschen Werte deine Variablen denn haben kannst du in der Zeile des Fehlers (wird dir im output angezeigt) einen Breakpoint setzen (links bei den Zeilennummern einfach draufklicken -> roter Punkt erscheint in der Zeile. Breakpoints halten immer noch vor der Zeile an)

Dann führst du den Test mit dem Debugger aus (rechtsklick in test-file -> Debug ...). Der Debugger hält beim Breakpoint an und zeigt dir die aktuellen Variablenwerte.
