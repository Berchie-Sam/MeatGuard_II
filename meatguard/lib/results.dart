import 'package:flutter/material.dart';

// -----------------------------------------------------------------------------
// RESULTS SCREEN (results.dart)
// -----------------------------------------------------------------------------
class ResultsScreen extends StatelessWidget {
  const ResultsScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9FAFB),
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(context),

            Expanded(
              child: SingleChildScrollView(
                physics: const BouncingScrollPhysics(),
                padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildMetricsCard(context),
                    const SizedBox(height: 16),

                    _buildHeatmapCard(context),
                    const SizedBox(height: 16),

                    _buildMetaCard(context),
                    const SizedBox(height: 24),

                    _buildActionButtonsRow(context),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAppBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Back button inside rounded circular container
          InkWell(
            onTap: () => Navigator.pushReplacementNamed(context, '/scan'),
            borderRadius: BorderRadius.circular(24),
            child: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
                border: Border.all(color: Colors.grey.shade200),
                boxShadow: [
                  BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 4),
                ],
              ),
              child: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.black87, size: 18),
            ),
          ),

          // Brand Title
          Column(
            children: [
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding: const EdgeInsets.all(4),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFECEE),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: const Icon(Icons.shield_rounded, color: Color(0xFFE53935), size: 16),
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'MeatGuard',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF2D3748),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 2),
              Text(
                'Smart Meat Freshness Detection',
                style: TextStyle(
                  fontSize: 11,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),

          const SizedBox(width: 40), // Balance spacing
        ],
      ),
    );
  }

  Widget _buildMetricsCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          _buildMetricRow(
            icon: Icons.color_lens_outlined,
            iconColor: const Color(0xFF43A047),
            iconBgColor: const Color(0xFFE8F5E9),
            title: 'Color',
            subtitle: 'Bright red color detected',
            status: 'Excellent',
            statusColor: const Color(0xFF43A047),
            progress: 0.92,
          ),
          const Divider(height: 24),
          _buildMetricRow(
            icon: Icons.psychology_outlined,
            iconColor: const Color(0xFF43A047),
            iconBgColor: const Color(0xFFE8F5E9),
            title: 'Texture',
            subtitle: 'Firm and elastic texture',
            status: 'Excellent',
            statusColor: const Color(0xFF43A047),
            progress: 0.88,
          ),
          const Divider(height: 24),
          _buildMetricRow(
            icon: Icons.water_drop_outlined,
            iconColor: const Color(0xFF43A047),
            iconBgColor: const Color(0xFFE8F5E9),
            title: 'Moisture',
            subtitle: 'Optimal moisture level',
            status: 'Good',
            statusColor: const Color(0xFF43A047),
            progress: 0.80,
          ),
          const Divider(height: 24),
          _buildMetricRow(
            icon: Icons.grain_rounded,
            iconColor: const Color(0xFF43A047),
            iconBgColor: const Color(0xFFE8F5E9),
            title: 'Marbling',
            subtitle: 'Healthy fat distribution',
            status: 'Excellent',
            statusColor: const Color(0xFF43A047),
            progress: 0.90,
          ),
        ],
      ),
    );
  }

  Widget _buildMetricRow({
    required IconData icon,
    required Color iconColor,
    required Color iconBgColor,
    required String title,
    required String subtitle,
    required String status,
    required Color statusColor,
    required double progress,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: iconBgColor,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(icon, color: iconColor, size: 20),
                ),
                const SizedBox(width: 12),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2D3748),
                  ),
                ),
              ],
            ),
            Text(
              status,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.bold,
                color: statusColor,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Padding(
          padding: const EdgeInsets.only(left: 40),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                subtitle,
                style: TextStyle(fontSize: 12, color: Colors.grey[600]),
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: progress,
                  backgroundColor: Colors.grey.shade100,
                  color: const Color(0xFF43A047),
                  minHeight: 6,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildHeatmapCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'AI Analysis Heatmap',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF2D3748),
                ),
              ),
              InkWell(
                onTap: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('About Heatmap'),
                      content: const Text('The AI heatmap highlights key regions analyzed by our TFLite model to evaluate freshness confidence.'),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Got it', style: TextStyle(color: Color(0xFFE53935))),
                        ),
                      ],
                    ),
                  );
                },
                child: Row(
                  children: [
                    Text('What is this?', style: TextStyle(fontSize: 12, color: Colors.grey[700], fontWeight: FontWeight.w500)),
                    const SizedBox(width: 4),
                    Icon(Icons.info_outline_rounded, color: Colors.grey[600], size: 16),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Heatmap image mockup container
              Container(
                width: 140,
                height: 95,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  gradient: const LinearGradient(
                    colors: [Color(0xFF0D47A1), Color(0xFF00ACC1), Color(0xFFFFEB3B), Color(0xFFE53935)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  boxShadow: [
                    BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 4),
                  ],
                ),
                child: Stack(
                  children: [
                    Center(
                      child: Container(
                        width: 50,
                        height: 50,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.white.withOpacity(0.2),
                        ),
                      ),
                    ),
                    const Center(
                      child: Icon(Icons.hub_rounded, color: Colors.white, size: 28),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'The heatmap shows areas the AI focused on to make its prediction.',
                      style: TextStyle(fontSize: 12, color: Colors.grey[700], height: 1.4),
                    ),
                    const SizedBox(height: 10),
                    // Color scale bar
                    Container(
                      height: 8,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(4),
                        gradient: const LinearGradient(
                          colors: [Color(0xFF2196F3), Color(0xFF4CAF50), Color(0xFFFFEB3B), Color(0xFFE53935)],
                        ),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Low Focus', style: TextStyle(fontSize: 10, color: Colors.grey[500])),
                        Text('High Focus', style: TextStyle(fontSize: 10, color: Colors.grey[500])),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetaCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade200),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade100,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.calendar_today_outlined, color: Colors.black87, size: 20),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('DATE & TIME', style: TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: Colors.grey[500], letterSpacing: 0.5)),
                      const SizedBox(height: 2),
                      const Text('20 May 2025, 08:32...', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF2D3748)), overflow: TextOverflow.ellipsis),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Container(height: 30, width: 1, color: Colors.grey.shade200),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(left: 12),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.location_on_outlined, color: Colors.black87, size: 20),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('LOCATION', style: TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: Colors.grey[500], letterSpacing: 0.5)),
                        const SizedBox(height: 2),
                        const Text('Kumasi Central Mar...', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF2D3748)), overflow: TextOverflow.ellipsis),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtonsRow(BuildContext context) {
    return Row(
      children: [
        // Rescan Button
        Expanded(
          child: InkWell(
            onTap: () => Navigator.pushReplacementNamed(context, '/scan'),
            borderRadius: BorderRadius.circular(16),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Icon(Icons.refresh_rounded, color: Colors.black87, size: 20),
                  SizedBox(height: 4),
                  Text('Rescan', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF2D3748))),
                  Text('Scan again', style: TextStyle(fontSize: 9, color: Colors.grey)),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 10),

        // Discard Button
        Expanded(
          child: InkWell(
            onTap: () {
              Navigator.pushReplacementNamed(context, '/home');
            },
            borderRadius: BorderRadius.circular(16),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                color: const Color(0xFFFFECEE),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Icon(Icons.delete_outline_rounded, color: Color(0xFFE53935), size: 20),
                  SizedBox(height: 4),
                  Text('Discard', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFFE53935))),
                  Text('Don\'t save', style: TextStyle(fontSize: 9, color: Color(0xFFC62828))),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 10),

        // Save Result Button
        Expanded(
          flex: 1,
          child: InkWell(
            onTap: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Result successfully saved to history!')),
              );
              Navigator.pushReplacementNamed(context, '/history');
            },
            borderRadius: BorderRadius.circular(16),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                color: const Color(0xFFE8F5E9),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFC8E6C9)),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Icon(Icons.download_rounded, color: Color(0xFF43A047), size: 20),
                  SizedBox(height: 4),
                  Text('Save Result', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF2E7D32))),
                  Text('Save to history', style: TextStyle(fontSize: 9, color: Color(0xFF388E3C))),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}