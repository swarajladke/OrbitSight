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

# Table formatting, and keep each table whole via KeepWithNext on every
# paragraph in the table except the last one.
foreach ($table in $doc.Tables) {
    $table.AutoFitBehavior(2)
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
    $firstPage = $table.Rows.Item(1).Range.Information(3)
    $lastPage = $table.Rows.Item($table.Rows.Count).Range.Information(3)
    if ($firstPage -eq $lastPage) { $verdict = "OK" } else { $verdict = "SPLIT" }
    Write-Output "Table $tableIndex : FirstRowPage = $firstPage, LastRowPage = $lastPage -> $verdict"
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
