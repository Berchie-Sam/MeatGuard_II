import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History', style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold)),
        systemOverlayStyle: SystemUiOverlayStyle.dark,
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black87),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: const Center(
        child: Text('History Screen Content Will Go Here'),
      ),
    );
  }
}