$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open("C:\Users\Vicky\Desktop\OrbitSight\OrbitSight_Research\PROPOSAL.html")

# 0.8-inch margins (57.6 pt)
$doc.PageSetup.TopMargin = 57.6
$doc.PageSetup.BottomMargin = 57.6
$doc.PageSetup.LeftMargin = 57.6
$doc.PageSetup.RightMargin = 57.6

# Paragraph spacing
foreach ($p in $doc.Paragraphs) {
    if ($p.Range.Tables.Count -eq 0) {
        $p.SpaceBefore = 0
        $p.SpaceAfter = 3
        $p.LineSpacingRule = 0 # wdLineSpaceSingle
    }
}

# Usable text column width, for the fit gate below.
$textWidth = $doc.PageSetup.PageWidth - $doc.PageSetup.LeftMargin - $doc.PageSetup.RightMargin
Write-Output "TextWidth = $textWidth"

# Table formatting. Width is forced LAST, after every other table edit, so
# nothing can undo the autofit.
foreach ($table in $doc.Tables) {
    foreach ($row in $table.Rows) {
        $row.AllowBreakAcrossPages = 0
    }
    $table.Rows.Item(1).HeadingFormat = -1
    $table.Range.Font.Size = 9.5

    $paras = $table.Range.Paragraphs
    $n = $paras.Count
    for ($i = 1; $i -lt $n; $i++) {
        $paras.Item($i).KeepWithNext = $true
    }
    $paras.Item($n).KeepWithNext = $false

    $table.PreferredWidthType = 2
    $table.PreferredWidth = 100
    $table.AutoFitBehavior(2)
}

# Keep each figure with the paragraph that precedes it and with its caption.
foreach ($s in $doc.InlineShapes) {
    $sp = $s.Range.Paragraphs.Item(1)
    $sp.KeepWithNext = $true
    $prev = $sp.Previous(1)
    if ($prev -ne $null) { $prev.KeepWithNext = $true }
}

$doc.Repaginate()

# InlineShape page check
$shapeIndex = 1
foreach ($s in $doc.InlineShapes) {
    $figPage = $s.Range.Information(3)
    Write-Output "InlineShape $shapeIndex : Page = $figPage, Width = $($s.Width), Height = $($s.Height)"
    $shapeIndex++
}

# Sound split verdict: measure the FIRST row and the LAST row separately.
# Information(3) reports the end of whatever range it is given, so the range
# must be a single row for the answer to mean anything.
$tableIndex = 1
foreach ($table in $doc.Tables) {
    $sum = 0
    try {
        foreach ($col in $table.Columns) { $sum += $col.Width }
    } catch {
        $sum = -1
    }
    $firstPage = $table.Rows.Item(1).Range.Information(3)
    $lastPage = $table.Rows.Item($table.Rows.Count).Range.Information(3)
    if ($firstPage -eq $lastPage) { $verdict = "OK" } else { $verdict = "SPLIT" }
    if ($sum -ge 0 -and $sum -le ($textWidth + 1)) { $fit = "FITS" } else { $fit = "OVERFLOW" }
    Write-Output "Table $tableIndex : Columns = $($table.Columns.Count), ColumnSum = $sum, FirstRowPage = $firstPage, LastRowPage = $lastPage -> $verdict / $fit"
    $tableIndex++
}

# wdFormatPDF = 17
$doc.SaveAs([ref]"C:\Users\Vicky\Desktop\OrbitSight\OrbitSight_Research\PROPOSAL.pdf", [ref]17)
$pages = $doc.ComputeStatistics(2)

if ($pages -gt 5) {
    $pageStart = $doc.GoTo(1, 1, $pages)
    $lastPageText = $pageStart.Paragraphs | ForEach-Object { $_.Range.Text }
    Write-Output "--- LAST PAGE TEXT ---"
    Write-Output ($lastPageText -join "")
}

$doc.Close([ref]$false)
$word.Quit()
Write-Output "FINAL_PAGE_COUNT: $pages"
