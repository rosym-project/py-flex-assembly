# Erstellen der Marker Annotationen

Im Folgenden wird erklärt, welche Schritte notwendig sind, um neue Marker für eine vorhandene Klemme zu erstellen.

## 1. Anlegen der Dateien
- Erstelle einen neuen Ordner `clamp_x_marker_i`
- Kopiere alle Dateien aus dem Ordner `unmarked` in den Ordner `clamp_x_marker_i`
- Bennene die Dateien zu `clamp_x_marker_i.*` um
- Passe den Namen und die beiden Mesh-Dateien in der SDF-Datei an

## 2. Einzeichnen des Markers in Blender
- Vorbereitung:
  - Lösche die beim Starten von Blender generierten Objekte: `a`, `del`
  - Importiere die Obj-Datei: `File > Import > Wavefront`
  - Vergrößere das Objekt (dies macht das markieren einfacher): `s`, `100`, `return` (die Maus dabei nicht bewegen)
  - Das Objekt durch anklicken auswählen (die orange Umrandung wird heller)
  - Auswahl der neuen Textur: `Material Properties` (rechts, zweiter Reiter von unten) > `Base Color` ausklappen > über das Ordner-Symbol `clamp_x_marker_i.png` auswählen
- Einzeichnen des Markers:
  - Oben den Modus `Texture Paint` auswählen. Nun ist links ein Fenster mit der Textur und rechts die gerenderte Klemme.
  - Anpassen des Pinsels:
    - Passe die RGB Farbe zu (1,0,0) an
    - Rechts sollte nun der Reiter `Active Tool and Workspace settings` aufgegangen sein
    - Unten unter `Falloff` das Rechteck auswählen.
    - Die Größe des Pinsels kann mit der Taste `f` und der Maus verändert werden (Bestätigen mit Linksklick)
  - Danach kann direkt auf dem Objekt gemalt werden
  - Bild speichern: linkes Fenster > `Image` > `Save`

## 3. Bestimmen der Markerposition
- Oben zurück zu `Layout` wechseln
- Skalierung rückgängig machen: `s`, `0,01`, `return`
- Wechsel in den Edit-Mode: `tabulator`
- Öffnen des Properties Panels: `n`
- Oben rechts wird der Median der ausgewählten Vertices angezeigt
- Zum Auswählen kann die orthografische Ansicht hilfreich sein: `1`, `3`, `7` auf dem Nummernblock, gleichzeitig `ctrl` zeigt die gegenüberliegende Seite an
- Alternativ:
  - In der Leiste links kann unter dem zweiten Punkt ein Cursor platziert werden
  - Um dessen Koordinaten anzuzeigen muss in dem Fenster oben rechts der Reiter `View` ausgewählt werden
  - Um den Cursor exakt zu positionieren kann die Quad-View hilfreich sein: `ctrl` + `alt` + `q`
- Der Dateipfad und die ermittelten Koordinaten müssen in die Datei `marker.csv` eingetragen werden
- Abschließend wird die Datei exportiert: `File` > `Export` > `Wavefront`, vorherige Datei überschreiben
