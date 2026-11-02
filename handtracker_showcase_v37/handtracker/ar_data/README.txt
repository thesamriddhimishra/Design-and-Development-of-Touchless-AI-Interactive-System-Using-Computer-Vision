Drop .csv files in this folder and they show up under the FILES button in AR mode.
(Re-scanned every time you open the Files row - no restart needed.)

FORMAT: two columns - a label, then a number. Header row optional.

    label,value
    Q1,120
    Q2,95

Also understood, so you can save straight from Excel/Sheets/Numbers:
  - comma, semicolon, tab or pipe delimiters (auto-detected)
  - UTF-8 or Windows (cp1252) text encoding
  - 1,200  /  1.200,50  /  1,5  /  $3.50  /  45%  style numbers
  - extra columns (ignored), blank lines, quoted labels containing commas
  - rows whose value isn't a number are skipped rather than failing the file

LIMITS:
  - Only the first 2 columns are used (one series per chart for now).
  - First 20 rows are shown; files over 5 MB are refused.
  - Negative values draw as an empty bar but their real number is still
    printed above it.

EDIT A FILE WHILE A CHART IS ON SCREEN, then hold TWO FINGERS over the
chart to reload it from disk.

If a file is broken, a [WARN] line appears in the console and the chart
keeps its last good data - it never crashes the app.
